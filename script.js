document.addEventListener("DOMContentLoaded", function () {
  const preloader = document.getElementById("preloader");
  if (preloader) {
    preloader.style.display = "none"; // Hide preloader after page load
  } else {
    console.error("Element with ID 'preloader' not found.");
  }

  const darkModeButton = document.getElementById("dark-mode-button");

  // Check if darkModeButton exists before proceeding
  if (darkModeButton) {
    // Dark mode and light mode functions
    const applyDarkMode = () => {
      document.body.classList.add("dark-mode");
      document.body.classList.remove("light-mode");
      darkModeButton.innerHTML = '<i class="fa-solid fa-sun"></i>'; // Set icon to sun
      document.body.style.backgroundColor = "rgba(50, 50, 50, 0.95)";
      document.body.style.color = "white";
    };

    const applyLightMode = () => {
      document.body.classList.add("light-mode");
      document.body.classList.remove("dark-mode");
      darkModeButton.innerHTML = '<i class="fa-solid fa-moon"></i>'; // Set icon to moon
      document.body.style.backgroundColor = ""; // Reset to default
      document.body.style.color = ""; // Reset to default
    };

    // Apply the current theme from sessionStorage on load
    const currentTheme = sessionStorage.getItem("theme");
    if (currentTheme === "dark") {
      applyDarkMode();
    } else {
      applyLightMode();
    }

    // Toggle dark mode on button click
    darkModeButton.addEventListener("click", () => {
      if (document.body.classList.contains("dark-mode")) {
        sessionStorage.setItem("theme", "light");
        applyLightMode();
      } else {
        sessionStorage.setItem("theme", "dark");
        applyDarkMode();
      }
    });
  } else {
    console.warn("Element with ID 'dark-mode-button' not found.");
  }
});

let script = document.createElement('script');
script.src = "https://cdn.gtranslate.net/widgets/latest/float.js"; // URL of the external script
script.defer = true; // Ensures the script runs after parsing the HTML
document.body.appendChild(script); // Add the script to the body

window.gtranslateSettings = { 
  "default_language": "en", 
  "detect_browser_language": true, 
  "wrapper_selector": ".gtranslate_wrapper" 
};

// Function to change the active class when a link is clicked
function changeContent(page) {
  const links = document.querySelectorAll(".menu ul li a");

  // Remove "active" class from all links
  links.forEach((link) => link.classList.remove("active"));

  // Add "active" class to the current page link
  const activeLink = document.getElementById(page + "-link");
  if (activeLink) {
    activeLink.classList.add("active");
  } else {
    console.error(`Link with id ${page + '-link'} not found`);
  }
}

// Language-based content visibility
const currentLanguage = window.gtranslateSettings.current_language;
const ambuFlowText = document.querySelector(".main_heading h2[data-link_h2='AmbuFlow...']");
if (currentLanguage === 'gu' || currentLanguage === 'hi') {
  ambuFlowText.style.display = 'none'; // Hide text for Gujarati and Hindi
} else {
  ambuFlowText.style.display = 'block'; // Show text for other languages
}

// Function that runs when the window loads
window.onload = function () {
  const currentPage = window.location.pathname.split("/").pop().replace(".html", "");
  if (currentPage) {
    changeContent(currentPage);
  }

  // Add delay to each letter drop animation
  const letters = document.querySelectorAll('.letter');
  letters.forEach((letter, index) => {
    letter.style.animationDelay = `${index * 0.1}s`;
  });
};

// Dark mode functionality


// Translator toggle functionality
const translateBtn = document.getElementById('translateBtn');
const gTranslate = document.getElementById('gTranslate');

translateBtn.addEventListener('click', () => {
  gTranslate.style.display = gTranslate.style.display === 'none' ? 'block' : 'none';
});

// Close the translator popup when clicking outside
document.addEventListener('click', (event) => {
  if (!gTranslate.contains(event.target) && event.target !== translateBtn) {
    gTranslate.style.display = 'none';
  }
});

// Newsletter form submission handler
document.getElementById("newsletter-form").addEventListener("submit", function (event) {
  event.preventDefault(); // Prevent the form from submitting normally
  const emailInput = document.getElementById("email");
  const confirmationMessage = document.getElementById("confirmation-message");

  const email = emailInput.value;
  console.log(`Email submitted: ${email}`); // For debugging
  confirmationMessage.textContent = "Thank you for subscribing! Please check your email for further instructions.";
  confirmationMessage.classList.remove("hidden");
  emailInput.value = ""; // Clear the form
});

// Accordion functionality
const accordions = document.querySelectorAll(".accordion");

accordions.forEach((accordion, index) => {
  const header = accordion.querySelector(".accordion__header");
  const content = accordion.querySelector(".accordion__content");
  const icon = accordion.querySelector(".accordion__icon i");

  header.addEventListener("click", () => {
    const isOpen = content.style.height === `${content.scrollHeight}px`;

    accordions.forEach((a, i) => {
      const c = a.querySelector(".accordion__content");
      const ic = a.querySelector(".accordion__icon i");

      if (i === index) {
        c.style.height = isOpen ? "0px" : `${c.scrollHeight}px`;
        ic.classList.toggle("ri-add-line", isOpen);
        ic.classList.toggle("ri-subtract-fill", !isOpen);
      } else {
        c.style.height = "0px";
        ic.classList.add("ri-add-line");
        ic.classList.remove("ri-subtract-fill");
      }
    });
  });
});

// Back to top button functionality
const backToTopButton = document.getElementById('back-to-top');

window.onscroll = function () {
  if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
    backToTopButton.style.display = "block";
  } else {
    backToTopButton.style.display = "none";
  }
};
