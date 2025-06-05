/**
 * Override the responsive navigation behavior to always show the navigation items
 * and remove the hamburger menu
 */
(function() {
  // Run after the DOM is fully loaded
  function init() {
    // Force the nav to stay open
    var navCollapse = document.querySelector('.nav-collapse');
    if (navCollapse) {
      // Remove closed class and add opened class
      navCollapse.classList.remove('closed');
      navCollapse.classList.add('opened');

      // Remove aria-hidden attribute to ensure accessibility
      navCollapse.removeAttribute('aria-hidden');

      // Make sure it has proper positioning
      navCollapse.style.position = 'relative';
      navCollapse.style.maxHeight = 'none';
      navCollapse.style.overflow = 'visible';

      // Completely remove the nav toggle button if it exists
      var navToggle = document.querySelector('.nav-toggle');
      if (navToggle && navToggle.parentNode) {
        navToggle.parentNode.removeChild(navToggle);
      }
    }
  }

  // Prevent errors when responsive-nav is not loaded
  if (typeof window.responsiveNav === 'undefined') {
    window.responsiveNav = function() {
      // Empty function to prevent errors
      console.log('Responsive nav not loaded, using fallback');
      return {
        toggle: function() {},
        open: function() {},
        close: function() {},
        destroy: function() {}
      };
    };
  }

  // Run immediately and also after the DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Also run after a slight delay to ensure everything is properly initialized
  setTimeout(init, 500);
})();
