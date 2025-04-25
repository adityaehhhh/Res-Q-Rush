// Cache DOM elements
const signInBtn = document.querySelector("#sign-in-btn");
const signUpBtn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");
const passwordField = document.querySelector(".sign-up-form input[type='password']");
const strengthWeak = document.getElementById("strength-weak");
const strengthMedium = document.getElementById("strength-medium");
const strengthStrong = document.getElementById("strength-strong");

// Toggle between Sign-In and Sign-Up modes
signUpBtn.addEventListener("click", () => {
  container.classList.add("sign-up-mode");
});

signInBtn.addEventListener("click", () => {
  container.classList.remove("sign-up-mode");
});

// Toggle password visibility
function togglePassword(fieldId, icon) {
  const field = document.getElementById(fieldId);
  const isPassword = field.type === "password";

  // Toggle between 'password' and 'text' types
  field.type = isPassword ? "text" : "password";

  // Toggle the icon class between 'eye' and 'eye-slash'
  icon.classList.toggle("fa-eye-slash", isPassword);
  icon.classList.toggle("fa-eye", !isPassword);

  // Set aria-label for accessibility
  icon.setAttribute("aria-label", isPassword ? "Hide password" : "Show password");
}

// Check password strength
function checkPasswordStrength() {
  const password = passwordField.value;
  let strength = 0;

  // Password strength criteria
  if (password.length >= 8) strength++;
  if (/[A-Z]/.test(password)) strength++;
  if (/[a-z]/.test(password)) strength++;
  if (/[0-9]/.test(password)) strength++;
  if (/[^a-zA-Z0-9]/.test(password)) strength++;

  // Reset classes
  strengthWeak.className = "";
  strengthMedium.className = "";
  strengthStrong.className = "";

  // Update strength indicators dynamically
  if (strength >= 1) strengthWeak.className = "weak";
  if (strength >= 3) strengthMedium.className = "medium";
  if (strength >= 5) strengthStrong.className = "strong";
  
  // Optional: Provide feedback to user
  displayStrengthMessage(strength);
}

// Display feedback message for password strength
function displayStrengthMessage(strength) {
  const feedbackMessage = document.getElementById("password-feedback");

  let message = "";
  let color = "";

  switch (strength) {
    case 0:
    case 1:
      message = "Weak password";
      color = "red";
      break;
    case 2:
    case 3:
      message = "Moderate password";
      color = "orange";
      break;
    case 4:
    case 5:
      message = "Strong password";
      color = "green";
      break;
    default:
      message = "";
  }

  feedbackMessage.textContent = message;
  feedbackMessage.style.color = color;
}

// Call the checkPasswordStrength function on password input
passwordField.addEventListener("input", checkPasswordStrength);