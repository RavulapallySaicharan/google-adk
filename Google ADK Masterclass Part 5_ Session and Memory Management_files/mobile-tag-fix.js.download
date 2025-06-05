/**
 * Simple fix for tag display as comma-separated list
 */
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    // Add a class to identify the tag containers that have been processed
    document.querySelectorAll('.tags-container').forEach(container => {
      container.classList.add('tags-processed');
    });
    
    // Make sure the body has a class that indicates JS is active
    document.body.classList.add('js-enabled');
    
    // Fix any excessive spacing in separators
    document.querySelectorAll('.tag-separator').forEach(sep => {
      if (sep.textContent.trim() === ',') {
        sep.textContent = ', ';
      }
    });
  });
})();
