"""Defensive chaos engineering for RepoRover.

Fault injection here is for resilience testing only: against RepoRover's own
in-process seams, and against local sandboxes the user owns. Nothing in this
package produces exploit code, and the target runner refuses any address that
is not local.
"""
