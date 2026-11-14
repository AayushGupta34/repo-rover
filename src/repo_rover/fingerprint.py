"""Deterministic stack fingerprinting.

Runs before any agent. Detects languages, frameworks and libraries from
dependency manifests, import statements, script tags and config files, and
records where each one was seen. No LLM is involved: the stack an analysis
run reports, and every `stack_trigger` it writes to the database, comes from
here.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repo_rover.repo_tools import RepoPathError, iter_repo_files, load_text_file

MAX_SOURCE_FILES_SCANNED = 600
MAX_SOURCE_CHARS = 200_000
MAX_EVIDENCE_PER_COMPONENT = 3
MAX_STACK_TRIGGER_COMPONENTS = 12

LANGUAGE_BY_EXTENSION = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".cjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".html": "HTML",
    ".htm": "HTML", ".css": "CSS", ".scss": "CSS", ".go": "Go", ".rs": "Rust",
    ".rb": "Ruby", ".php": "PHP", ".java": "Java", ".kt": "Kotlin", ".swift": "Swift",
    ".cs": "C#", ".sql": "SQL", ".vue": "Vue", ".svelte": "Svelte", ".dart": "Dart",
    ".sh": "Shell",
}
MARKUP_LANGUAGES = {"HTML", "CSS", "SQL", "Shell"}

# package name -> (canonical component, kind)
NPM_COMPONENTS = {
    "next": ("Next.js", "framework"), "react": ("React", "framework"),
    "react-dom": ("React", "framework"), "react-native": ("React Native", "framework"),
    "expo": ("Expo", "framework"), "vue": ("Vue", "framework"), "nuxt": ("Nuxt", "framework"),
    "svelte": ("Svelte", "framework"), "@sveltejs/kit": ("SvelteKit", "framework"),
    "@angular/core": ("Angular", "framework"), "express": ("Express", "framework"),
    "fastify": ("Fastify", "framework"), "koa": ("Koa", "framework"), "hono": ("Hono", "framework"),
    "@nestjs/core": ("NestJS", "framework"), "@remix-run/react": ("Remix", "framework"),
    "astro": ("Astro", "framework"), "electron": ("Electron", "framework"),
    "alpinejs": ("Alpine.js", "framework"), "htmx.org": ("htmx", "framework"),
    "tailwindcss": ("Tailwind CSS", "library"), "bootstrap": ("Bootstrap", "library"),
    "jquery": ("jQuery", "library"), "vite": ("Vite", "tool"), "webpack": ("webpack", "tool"),
    "typescript": ("TypeScript", "tool"), "prisma": ("Prisma", "library"),
    "@prisma/client": ("Prisma", "library"), "mongoose": ("Mongoose", "library"),
    "mongodb": ("MongoDB", "database"), "pg": ("PostgreSQL", "database"),
    "postgres": ("PostgreSQL", "database"), "mysql": ("MySQL", "database"),
    "mysql2": ("MySQL", "database"), "sqlite3": ("SQLite", "database"),
    "better-sqlite3": ("SQLite", "database"), "sequelize": ("Sequelize", "library"),
    "typeorm": ("TypeORM", "library"), "drizzle-orm": ("Drizzle ORM", "library"),
    "redis": ("Redis", "database"), "ioredis": ("Redis", "database"),
    "@supabase/supabase-js": ("Supabase", "service"), "firebase": ("Firebase", "service"),
    "firebase-admin": ("Firebase", "service"), "stripe": ("Stripe", "service"),
    "@stripe/stripe-js": ("Stripe", "service"), "razorpay": ("Razorpay", "service"),
    "aws-sdk": ("AWS SDK", "service"), "@aws-sdk/client-s3": ("AWS SDK", "service"),
    "jsonwebtoken": ("jsonwebtoken", "library"), "jose": ("jose", "library"),
    "next-auth": ("NextAuth", "library"), "@clerk/nextjs": ("Clerk", "service"),
    "passport": ("Passport", "library"), "bcrypt": ("bcrypt", "library"),
    "bcryptjs": ("bcrypt", "library"), "express-session": ("express-session", "library"),
    "cookie-session": ("cookie-session", "library"), "multer": ("multer", "library"),
    "axios": ("axios", "library"), "graphql": ("GraphQL", "library"),
    "@apollo/server": ("Apollo Server", "library"), "apollo-server": ("Apollo Server", "library"),
    "socket.io": ("Socket.IO", "library"), "ws": ("WebSocket", "library"),
    "openai": ("OpenAI", "service"), "@anthropic-ai/sdk": ("Anthropic", "service"),
    "@google/generative-ai": ("Gemini", "service"), "groq-sdk": ("Groq", "service"),
    "langchain": ("LangChain", "library"), "ai": ("Vercel AI SDK", "library"),
    "zod": ("zod", "library"), "lodash": ("lodash", "library"), "moment": ("moment", "library"),
    "@tanstack/react-query": ("TanStack Query", "library"), "swr": ("SWR", "library"),
    "node-cron": ("node-cron", "library"), "nodemailer": ("nodemailer", "library"),
}

PYPI_COMPONENTS = {
    "django": ("Django", "framework"), "djangorestframework": ("Django REST Framework", "framework"),
    "flask": ("Flask", "framework"), "fastapi": ("FastAPI", "framework"),
    "starlette": ("Starlette", "framework"), "streamlit": ("Streamlit", "framework"),
    "gradio": ("Gradio", "framework"), "crewai": ("CrewAI", "framework"),
    "sqlalchemy": ("SQLAlchemy", "library"), "flask-sqlalchemy": ("SQLAlchemy", "library"),
    "sqlmodel": ("SQLModel", "library"), "alembic": ("Alembic", "library"),
    "psycopg2": ("PostgreSQL", "database"), "psycopg2-binary": ("PostgreSQL", "database"),
    "psycopg": ("PostgreSQL", "database"), "asyncpg": ("PostgreSQL", "database"),
    "pymysql": ("MySQL", "database"), "pymongo": ("MongoDB", "database"),
    "redis": ("Redis", "database"), "celery": ("Celery", "library"),
    "requests": ("requests", "library"), "httpx": ("httpx", "library"),
    "pyjwt": ("PyJWT", "library"), "python-jose": ("python-jose", "library"),
    "passlib": ("passlib", "library"), "bcrypt": ("bcrypt", "library"),
    "pydantic": ("Pydantic", "library"), "pyyaml": ("PyYAML", "library"),
    "boto3": ("AWS SDK", "service"), "stripe": ("Stripe", "service"),
    "supabase": ("Supabase", "service"), "firebase-admin": ("Firebase", "service"),
    "openai": ("OpenAI", "service"), "anthropic": ("Anthropic", "service"),
    "groq": ("Groq", "service"), "google-generativeai": ("Gemini", "service"),
    "langchain": ("LangChain", "library"), "langchain-core": ("LangChain", "library"),
    "langchain-openai": ("LangChain", "library"), "langchain-groq": ("LangChain", "library"),
    "uvicorn": ("Uvicorn", "tool"), "gunicorn": ("Gunicorn", "tool"),
    "pandas": ("pandas", "library"), "numpy": ("NumPy", "library"),
    "apscheduler": ("APScheduler", "library"), "python-dotenv": ("python-dotenv", "library"),
}

# Python import name -> PyPI package name, where they differ.
PYTHON_IMPORT_ALIASES = {
    "jwt": "pyjwt", "yaml": "pyyaml", "rest_framework": "djangorestframework",
    "flask_sqlalchemy": "flask-sqlalchemy", "jose": "python-jose", "dotenv": "python-dotenv",
    "firebase_admin": "firebase-admin", "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai", "langchain_groq": "langchain-groq",
    "google": "google-generativeai",
}
PYTHON_STDLIB_COMPONENTS = {"sqlite3": ("SQLite", "database")}

# substring of a script/link URL -> component
CDN_COMPONENTS = (
    ("tailwindcss", "Tailwind CSS", "library"), ("alpinejs", "Alpine.js", "framework"),
    ("alpine.js", "Alpine.js", "framework"), ("vue", "Vue", "framework"),
    ("react-dom", "React", "framework"), ("react", "React", "framework"),
    ("jquery", "jQuery", "library"), ("bootstrap", "Bootstrap", "library"),
    ("htmx", "htmx", "framework"), ("chart.js", "Chart.js", "library"),
    ("chartjs", "Chart.js", "library"), ("axios", "axios", "library"),
    ("lodash", "lodash", "library"), ("firebase", "Firebase", "service"),
    ("supabase", "Supabase", "service"), ("stripe", "Stripe", "service"),
    ("socket.io", "Socket.IO", "library"), ("d3", "D3", "library"),
)

# (file-name regex, component, kind)
CONFIG_FILE_COMPONENTS = (
    (r"^next\.config\.(js|mjs|ts)$", "Next.js", "framework"),
    (r"^nuxt\.config\.(js|ts)$", "Nuxt", "framework"),
    (r"^vite\.config\.(js|mjs|ts)$", "Vite", "tool"),
    (r"^svelte\.config\.(js|ts)$", "Svelte", "framework"),
    (r"^angular\.json$", "Angular", "framework"),
    (r"^tailwind\.config\.(js|cjs|mjs|ts)$", "Tailwind CSS", "library"),
    (r"^tsconfig(\..+)?\.json$", "TypeScript", "tool"),
    (r"^(dockerfile(\..+)?|.+\.dockerfile)$", "Docker", "tool"),
    (r"^(docker-)?compose(\..+)?\.ya?ml$", "Docker Compose", "tool"),
    (r"^firebase\.json$", "Firebase", "service"),
    (r"^(firestore|storage|database)\.rules$", "Firebase", "service"),
    (r"^vercel\.json$", "Vercel", "service"),
    (r"^netlify\.toml$", "Netlify", "service"),
    (r"^schema\.prisma$", "Prisma", "library"),
    (r"^manage\.py$", "Django", "framework"),
    (r"^.+\.tf$", "Terraform", "tool"),
)

# Web platform APIs worth knowing about when choosing detectors and scenarios.
PLATFORM_API_PATTERNS = (
    (re.compile(r"\blocalStorage\."), "localStorage"),
    (re.compile(r"\bindexedDB\b"), "IndexedDB"),
    (re.compile(r"\bnew\s+WebSocket\("), "WebSocket"),
    (re.compile(r"navigator\.serviceWorker"), "Service Worker"),
)

FRAMEWORK_PRIORITY = (
    "Next.js", "Nuxt", "SvelteKit", "Remix", "Astro", "NestJS", "Django", "FastAPI",
    "Flask", "Express", "Fastify", "Koa", "Hono", "React Native", "Expo", "Electron",
    "Angular", "React", "Vue", "Svelte", "Streamlit", "Gradio", "CrewAI", "Alpine.js", "htmx",
)
BACKEND_FRAMEWORKS = frozenset({
    "Next.js", "Nuxt", "SvelteKit", "Remix", "NestJS", "Django", "Django REST Framework", "FastAPI",
    "Flask", "Starlette", "Express", "Fastify", "Koa", "Hono", "Streamlit", "Gradio",
})
FRONTEND_FRAMEWORKS = frozenset({"React", "Vue", "Svelte", "Angular", "Alpine.js", "htmx", "React Native", "Expo", "Astro"})
SOURCE_LANGUAGES = frozenset({"Python", "JavaScript", "TypeScript", "Go", "Rust", "Ruby", "PHP", "Java", "Kotlin",
                              "Swift", "C#", "Dart", "Vue", "Svelte", "HTML"})
KIND_ORDER = {"framework": 0, "database": 1, "service": 2, "library": 3, "platform_api": 4, "tool": 5}

_JS_IMPORT = re.compile(r"""(?:\bfrom\s+|\brequire\(\s*|\bimport\s+|\bimport\(\s*)["']([^"'\s]+)["']""")
_PY_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.MULTILINE)
_SCRIPT_SRC = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["']""", re.IGNORECASE)
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_package(name: str) -> str:
    return re.sub(r"[_.]+", "-", name.strip().lower())


def _term_key(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", text.lower())


@dataclass
class Component:
    name: str
    kind: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Fingerprint:
    root: str
    languages: dict[str, int] = field(default_factory=dict)
    components: dict[str, Component] = field(default_factory=dict)
    # manifest path -> {"kind": file name, "packages": {normalized name: line}}
    manifests: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, name: str, kind: str, source: str, file: str, line: int, snippet: str) -> None:
        component = self.components.setdefault(name, Component(name, kind))
        if KIND_ORDER.get(kind, 9) < KIND_ORDER.get(component.kind, 9):
            component.kind = kind
        if len(component.evidence) < MAX_EVIDENCE_PER_COMPONENT and not any(
            e["source"] == source and e["file"] == file for e in component.evidence
        ):
            component.evidence.append(
                {"source": source, "file": file, "line": line, "snippet": snippet.strip()[:160]}
            )

    @property
    def primary_language(self) -> str:
        code = {k: v for k, v in self.languages.items() if k not in MARKUP_LANGUAGES}
        pool = code or self.languages
        return max(pool, key=lambda k: (pool[k], k)) if pool else "unknown"

    @property
    def has_source_code(self) -> bool:
        return any(language in SOURCE_LANGUAGES for language in self.languages)

    @property
    def archetype(self) -> str:
        """backend, frontend, static_site or library. Used to gate rules that
        only make sense for one kind of repository."""
        names = set(self.components)
        if names & BACKEND_FRAMEWORKS:
            return "backend"
        server_languages = {"Python", "Go", "Rust", "Ruby", "PHP", "Java", "Kotlin", "C#"}
        if "HTML" in self.languages and not self.manifests and not (set(self.languages) & server_languages):
            return "static_site"
        if names & FRONTEND_FRAMEWORKS:
            return "frontend"
        return "library"

    @property
    def primary_framework(self) -> str:
        for name in FRAMEWORK_PRIORITY:
            if name in self.components:
                return name
        return "none"

    def component_names(self) -> list[str]:
        return [
            c.name
            for c in sorted(
                self.components.values(),
                key=lambda c: (KIND_ORDER.get(c.kind, 9), -len(c.evidence), c.name.lower()),
            )
        ]

    @property
    def stack_trigger(self) -> str:
        """The stack as a comma-separated string, in the catalogue's format."""
        names = [n for n in self.component_names() if self.components[n].kind != "tool"]
        if not names:
            names = sorted(k for k in self.languages if k not in MARKUP_LANGUAGES) or sorted(self.languages)
        return ", ".join(names[:MAX_STACK_TRIGGER_COMPONENTS]) or "unknown"

    def allowed_terms(self) -> dict[str, str]:
        """Lookup of every term an agent may use for the stack -> canonical name."""
        terms = {_term_key(name): name for name in self.components}
        for language in self.languages:
            terms.setdefault(_term_key(language), language)
        return terms

    def validate_stack_trigger(self, text: Any) -> tuple[str | None, list[str]]:
        """Canonicalise an agent-supplied stack string against the fingerprint.

        Returns (canonical string, []) when every comma-separated term names a
        fingerprinted component or language, else (None, unknown terms).
        """
        allowed = self.allowed_terms()
        parts = [p.strip() for p in re.split(r"[,;/|]+|\s+and\s+|\s+\+\s+", str(text or "")) if p.strip()]
        if not parts:
            return None, ["(empty)"]
        canonical: list[str] = []
        unknown: list[str] = []
        for part in parts:
            match = allowed.get(_term_key(part))
            if match is None:
                unknown.append(part)
            elif match not in canonical:
                canonical.append(match)
        if unknown:
            return None, unknown
        return ", ".join(canonical), []

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_language": self.primary_language,
            "primary_framework": self.primary_framework,
            "archetype": self.archetype,
            "stack_trigger": self.stack_trigger,
            "languages": dict(sorted(self.languages.items(), key=lambda kv: -kv[1])),
            "components": [
                {"name": c.name, "kind": c.kind, "evidence": c.evidence}
                for c in (self.components[n] for n in self.component_names())
            ],
            "manifests": sorted(self.manifests),
        }

    def summary_text(self) -> str:
        """Compact, evidence-bearing description for prompts."""
        lines = [
            f"Primary language: {self.primary_language}",
            f"Primary framework: {self.primary_framework}",
            f"Archetype: {self.archetype}",
            "Languages (files): "
            + (", ".join(f"{k} {v}" for k, v in sorted(self.languages.items(), key=lambda kv: -kv[1])) or "none"),
            "Components (each proven by the file shown):",
        ]
        for name in self.component_names():
            component = self.components[name]
            first = component.evidence[0]
            lines.append(
                f"- {name} [{component.kind}] via {first['source']} at {first['file']}:{first['line']}"
            )
        if not self.components:
            lines.append("- none detected")
        return "\n".join(lines)


def _line_of(lines: list[str], needle: str) -> int:
    for number, line in enumerate(lines, start=1):
        if needle in line:
            return number
    return 1


def _parse_package_json(lines: list[str]) -> dict[str, int]:
    try:
        data = json.loads("\n".join(lines))
    except (json.JSONDecodeError, RecursionError, MemoryError):
        return {}
    packages: dict[str, int] = {}
    if not isinstance(data, dict):
        return packages
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        deps = data.get(section)
        if isinstance(deps, dict):
            for name in deps:
                packages.setdefault(str(name).lower(), _line_of(lines, f'"{name}"'))
    return packages


def _parse_requirements(lines: list[str]) -> dict[str, int]:
    packages: dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith(("-", "git+", "http")):
            continue
        match = _REQUIREMENT_NAME.match(stripped)
        if match:
            packages.setdefault(normalize_package(match.group(1)), number)
    return packages


def _requirement_names(specs: Any) -> list[str]:
    names = []
    if isinstance(specs, list):
        for spec in specs:
            match = _REQUIREMENT_NAME.match(spec) if isinstance(spec, str) else None
            if match:
                names.append(normalize_package(match.group(1)))
    elif isinstance(specs, dict):
        names.extend(normalize_package(str(key)) for key in specs)
    return names


def _parse_toml_manifest(lines: list[str]) -> dict[str, int]:
    try:
        data = tomllib.loads("\n".join(lines))
    except (tomllib.TOMLDecodeError, ValueError, RecursionError, MemoryError):
        return {}
    names: list[str] = []
    project = data.get("project", {}) if isinstance(data.get("project"), dict) else {}
    names += _requirement_names(project.get("dependencies"))
    for group in (project.get("optional-dependencies") or {}).values():
        names += _requirement_names(group)
    for group in (data.get("dependency-groups") or {}).values():
        names += _requirement_names(group)
    poetry = (data.get("tool") or {}).get("poetry") or {}
    names += _requirement_names(poetry.get("dependencies"))
    names += _requirement_names(poetry.get("dev-dependencies"))
    for key in ("packages", "dev-packages", "dependencies", "dev-dependencies"):  # Pipfile, Cargo.toml
        names += _requirement_names(data.get(key) if isinstance(data.get(key), dict) else None)
    return {name: _line_of(lines, name.split("-")[0]) for name in names if name != "python"}


MANIFEST_PARSERS = {
    "package.json": (_parse_package_json, NPM_COMPONENTS),
    "pyproject.toml": (_parse_toml_manifest, PYPI_COMPONENTS),
    "pipfile": (_parse_toml_manifest, PYPI_COMPONENTS),
}


def _js_package(specifier: str) -> str | None:
    if specifier.startswith((".", "/", "node:", "#", "~", "@/")):
        return None
    parts = specifier.split("/")
    name = "/".join(parts[:2]) if specifier.startswith("@") and len(parts) > 1 else parts[0]
    return name.lower()


def fingerprint_repo(root: str | Path) -> Fingerprint:
    """Scan a repository and return its deterministic stack fingerprint."""
    import os

    root = Path(os.path.realpath(root))
    fp = Fingerprint(root=str(root))
    scanned = 0

    for relative in iter_repo_files(root):
        name = relative.rsplit("/", 1)[-1]
        lowered = name.lower()
        extension = "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
        language = LANGUAGE_BY_EXTENSION.get(extension)
        if language:
            fp.languages[language] = fp.languages.get(language, 0) + 1

        for pattern, component, kind in CONFIG_FILE_COMPONENTS:
            if component and re.match(pattern, lowered):
                fp.add(component, kind, "config_file", relative, 1, name)
        if relative.startswith(".github/workflows/") and lowered.endswith((".yml", ".yaml")):
            fp.add("GitHub Actions", "tool", "config_file", relative, 1, name)

        is_manifest = lowered in MANIFEST_PARSERS or (
            lowered.startswith("requirements") and lowered.endswith(".txt")
        )
        is_source = extension in (".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte")
        is_markup = extension in (".html", ".htm", ".vue", ".svelte")
        if not (is_manifest or is_source or is_markup):
            continue
        if not is_manifest:
            scanned += 1
            if scanned > MAX_SOURCE_FILES_SCANNED:
                continue
        try:
            _, lines = load_text_file(root / relative)
        except RepoPathError:
            continue

        if is_manifest:
            if lowered in MANIFEST_PARSERS:
                parser, table = MANIFEST_PARSERS[lowered]
                packages = parser(lines)
            else:
                packages, table = _parse_requirements(lines), PYPI_COMPONENTS
            fp.manifests[relative] = {"kind": name, "packages": packages}
            for package, line in packages.items():
                if package in table:
                    component, kind = table[package]
                    fp.add(component, kind, "manifest", relative, line, lines[line - 1] if lines else package)
            continue

        text = "\n".join(lines)[:MAX_SOURCE_CHARS]

        if extension == ".py":
            for match in _PY_IMPORT.finditer(text):
                module = match.group(1)
                package = PYTHON_IMPORT_ALIASES.get(module, normalize_package(module))
                hit = PYPI_COMPONENTS.get(package) or PYTHON_STDLIB_COMPONENTS.get(module)
                if hit:
                    line = text.count("\n", 0, match.start()) + 1
                    fp.add(hit[0], hit[1], "import", relative, line, lines[line - 1])
        elif is_source:
            for match in _JS_IMPORT.finditer(text):
                package = _js_package(match.group(1))
                if package and package in NPM_COMPONENTS:
                    component, kind = NPM_COMPONENTS[package]
                    line = text.count("\n", 0, match.start()) + 1
                    fp.add(component, kind, "import", relative, line, lines[line - 1])

        if is_markup:
            for match in _SCRIPT_SRC.finditer(text):
                url = match.group(1).lower()
                if not url.startswith(("http://", "https://", "//")):
                    continue
                for needle, component, kind in CDN_COMPONENTS:
                    if re.search(rf"(^|[/@.\-_]){re.escape(needle)}([/@.\-_]|$)", url):
                        line = text.count("\n", 0, match.start()) + 1
                        fp.add(component, kind, "script_tag", relative, line, lines[line - 1])
                        break

        if is_markup or extension in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
            for pattern, component in PLATFORM_API_PATTERNS:
                match = pattern.search(text)
                if match:
                    line = text.count("\n", 0, match.start()) + 1
                    fp.add(component, "platform_api", "api_usage", relative, line, lines[line - 1])

    return fp
