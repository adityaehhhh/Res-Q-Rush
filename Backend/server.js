const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const bodyParser = require("body-parser");

const dotenv = require("dotenv");
dotenv.config();

const app = express();
const port = 3000;

// Middleware
app.use(bodyParser.json());

// MongoDB config below (Uncomment this code when mongoDB is created)

mongoose
  .connect(
    "your-connection-string",
    { useNewUrlParser: true, useUnifiedTopology: true }
  )
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.log(err));

// Models
const User = require("./Models/User");
const {
  createAmbulance,
  getAllAmbulances,
  bookAmbulance,
} = require("./Controllers/AmbulanceController");

app.get("/", (req, res) => {
  res.send("AmbuFlow Server");
});

// Signup
app.post("/signup", async (req, res) => {
  const { id, username, email, password } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    const user = new User({ id, username, email, password: hashedPassword });

    await user.save();

    res.status(201).json({
      message: "User created successfully",
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
      },
    });
  } catch (err) {
    res.status(500).json({ error: "Error creating user: " + err.message });
  }
});

// Login Route
app.post("/login", async (req, res) => {
  const { username, password } = req.body;

  // Uncomment the lines when MongoDB is created

  try {
    const user = await User.findOne({ username });

    if (!user) {
      return res.status(401).send("User not found");
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      return res.status(401).send("Invalid credentials");
    }

    const token = jwt.sign(
      { id: user.id, username: user.username },
      process.env.JWT_SECRET,
      { expiresIn: "1h" }
    );

    res.json({ success: true, JWT_token: token, id: user.id });
  } catch (err) {
    res.status(500).send("Internal Server Error: " + err.message);
  }

  // Hardcoded for testing purposes
  const token = jwt.sign(
    { id: 1, username: "username" },
    process.env.JWT_SECRET,
    { expiresIn: "1h" }
  );

  res.json({ success: true, JWT_token: token, id: "testID893973983" });
});

// Middleware to protect routes
const authenticateToken = (req, res, next) => {
  const token = req.headers["authorization"]?.split(" ")[1]; // Bearer token

  if (!token) {
    return res.sendStatus(401);
  }

  jwt.verify(token, JWT_secret, (err, user) => {
    if (err) {
      return res.sendStatus(403);
    }

    req.user = user;
    next();
  });
};

app.get("/protected", authenticateToken, (req, res) => {
  res.send("This is a protected route");
});

app.post("/ambulances/create", createAmbulance);
app.get("/ambulances", getAllAmbulances);
app.post("/ambulances/book", bookAmbulance);

app.listen(port, () => {
  console.log(`Example app listening on http://localhost:${port}`);
});
