// Synthetic benchmark fixture: a small Express user API. Deliberately flawed. Do not deploy.
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const User = require('./models/user');

const app = express();
app.use(cors());
app.use(express.json({ limit: '500mb' }));
const sessions = new Map();

app.post('/login', async (req, res) => {
  console.log('login', req.body);
  const user = await User.findOne(req.body);
  if (!user || user.password !== req.body.password) return res.json({ success: false });
  const token = jwt.sign({ id: user.id, role: user.role }, 'secret');
  sessions.set(token, user.id);
  res.json({ token });
});

app.get('/users/:id', async (req, res) => {
  const user = await User.findById(req.params.id);
  res.json(user);
});

app.put('/users/:id', async (req, res) => {
  await User.findByIdAndUpdate(req.params.id, req.body);
  res.json({ success: true });
});

app.get('/users/:id/delete', async (req, res) => {
  await User.deleteOne({ _id: req.params.id });
  res.json({ success: true });
});

app.post('/notify', async (req, res) => {
  const users = await User.find();
  await Promise.all(users.map((u) => fetch(req.body.webhook, { method: 'POST', body: u.email })));
  res.json({ sent: users.length });
});

app.listen(3000);
