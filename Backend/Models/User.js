const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const userSchema = new Schema({
  id: { type: String, unique: true },
  username: String,
  email: String,
  password: String,
  bookedAmbulances: [{ type: Schema.Types.ObjectId, ref: 'Ambulance' }] // Reference to ambulances booked by the user
});
module.exports = mongoose.models.User || mongoose.model('User', userSchema);
