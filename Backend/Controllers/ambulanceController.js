const Ambulance = require("../Models/Ambulance");
const User = require("../Models/User");


// Controller to get all ambulances
const getAllAmbulances = async (req, res) => {
  try {
    const ambulances = await Ambulance.find();
    res.json(ambulances);
  } catch (error) {
    res.status(500).json({ error: 'Server error, unable to fetch ambulances' });
  }
};

// Controller to book an ambulance by a user
const bookAmbulance = async (req, res) => {
  const { userId, ambulanceId } = req.body;

  try {
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    const ambulance = await Ambulance.findById(ambulanceId);
    if (!ambulance) {
      return res.status(404).json({ error: 'Ambulance not found' });
    }

    if (!ambulance.availabilityStatus) {
      return res.status(400).json({ error: 'Ambulance is already booked' });
    }

    ambulance.availabilityStatus = false;
    ambulance.user = user._id;
    await ambulance.save();

    user.bookedAmbulances.push(ambulance._id);
    await user.save();

    res.json({ message: 'Ambulance booked successfully', ambulance });
  } catch (error) {
    res.status(500).json({ error: 'Server error, unable to book ambulance' });
  }
};

// Controller to create a new ambulance
const createAmbulance = async (req, res) => {
  const { numberPlate, driverName, location } = req.body;

  try {
    const newAmbulance = new Ambulance({
      numberPlate,
      driverName,
      location,
      availabilityStatus: true, // By default, new ambulances are available
    });

    await newAmbulance.save();

    res.json({ message: 'Ambulance created successfully', ambulance: newAmbulance });
  } catch (error) {
    res.status(500).json({ error: 'Server error, unable to create ambulance' });
  }
};

module.exports = {
  getAllAmbulances,
  bookAmbulance,
  createAmbulance,
};
