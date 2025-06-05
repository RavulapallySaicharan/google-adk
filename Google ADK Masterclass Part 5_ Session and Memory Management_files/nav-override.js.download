/**
 * Override the responsive navigation initialization
 * to prevent the hamburger menu from ever being created
 */
(function() {
  // Check if responsive nav exists
  if (typeof window.responsiveNav === 'undefined') {
    // Create a dummy function to prevent errors
    window.responsiveNav = function(selector, options) {
      console.log('Responsive nav not loaded, using fallback');
      return {
        toggle: function() {},
        open: function() {},
        close: function() {},
        destroy: function() {}
      };
    };
    return; // Exit early
  }

  // Original responsiveNav function reference
  var originalResponsiveNav = window.responsiveNav;

  // Override the function
  window.responsiveNav = function(selector, options) {
    // Check if the selector exists in the DOM
    var navElement = document.querySelector(selector);
    if (!navElement) {
      console.log('Nav element not found:', selector);
      return {
        toggle: function() {},
        open: function() {},
        close: function() {},
        destroy: function() {}
      };
    }

    // Add our custom options to disable the hamburger
    var enhancedOptions = options || {};

    // Force no custom toggle
    enhancedOptions.customToggle = "";

    // Prevent creation of toggle button
    enhancedOptions._createToggle = function() {
      // Do nothing - don't create the toggle
    };

    try {
      // Call the original function with our modified options
      var navInstance = originalResponsiveNav(selector, enhancedOptions);

      // Force the nav to be always open
      setTimeout(function() {
        // Remove any existing nav toggle
        var existingToggle = document.querySelector('.nav-toggle');
        if (existingToggle && existingToggle.parentNode) {
          existingToggle.parentNode.removeChild(existingToggle);
        }

        // Open the navigation and keep it open
        if (navInstance) {
          navInstance.open();

          // Override the toggle method to prevent closing
          navInstance.toggle = function() {
            // Always ensure it's open
            navInstance.open();
          };
        }
      }, 100);

      return navInstance;
    } catch (e) {
      console.log('Error initializing responsive nav:', e.message);
      return {
        toggle: function() {},
        open: function() {},
        close: function() {},
        destroy: function() {}
      };
    }
  };
})();
