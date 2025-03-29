const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const ambulanceSchema = new Schema({
  numberPlate: { type: String, required: true, unique: true },
  driverName: { type: String, required: true },
  location: { type: String, required: true },
  availabilityStatus: { type: Boolean, default: true }, // Indicates if the ambulance is available
  user: { type: Schema.Types.ObjectId, ref: 'User', required: false } // Reference to the user who booked the ambulance
});

module.exports = mongoose.model('Ambulance', ambulanceSchema);
