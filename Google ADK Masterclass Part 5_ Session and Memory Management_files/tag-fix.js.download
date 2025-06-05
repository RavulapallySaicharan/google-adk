/**
 * Fix for blog post tags display
 */
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    // Target the tags container
    const tagContainers = document.querySelectorAll('.tags-container');
    
    tagContainers.forEach(container => {
      // Ensure the container stays inline with nowrap
      container.style.display = 'inline';
      container.style.whiteSpace = 'nowrap';
      
      // Fix all tag links inside the container
      const tagLinks = container.querySelectorAll('.tag-link');
      tagLinks.forEach(link => {
        link.style.display = 'inline';
        link.style.whiteSpace = 'nowrap';
      });
      
      // Fix the commas
      const separators = container.querySelectorAll('.tag-separator');
      separators.forEach(separator => {
        separator.style.display = 'inline';
        separator.style.whiteSpace = 'nowrap';
        separator.style.marginRight = '0.2rem';
      });
    });
    
    // Fix parent containers
    const metadataContainers = document.querySelectorAll('.mr1.nowrap');
    metadataContainers.forEach(container => {
      if (container.querySelector('.fa-tags')) {
        // This is a tag container
        container.style.whiteSpace = 'nowrap';
        container.style.overflow = 'visible';
      }
    });
  });
})();
