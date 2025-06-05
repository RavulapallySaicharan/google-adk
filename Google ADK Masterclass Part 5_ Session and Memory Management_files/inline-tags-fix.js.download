/**
 * Enhanced fix for blog post tags display
 */
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    // Target the tags container
    const tagContainers = document.querySelectorAll('.tags-container');
    
    // Process each tags container
    tagContainers.forEach(container => {
      // Get all the comma text nodes and make sure they're inline
      const textNodes = Array.from(container.childNodes).filter(node => 
        node.nodeType === Node.TEXT_NODE && node.textContent.includes(',')
      );
      
      textNodes.forEach(textNode => {
        const span = document.createElement('span');
        span.className = 'tag-separator';
        span.textContent = textNode.textContent;
        span.style.display = 'inline';
        span.style.whiteSpace = 'nowrap';
        textNode.parentNode.replaceChild(span, textNode);
      });
      
      // Fix all tag links to be inline-block
      const tagLinks = container.querySelectorAll('.tag-link');
      tagLinks.forEach(link => {
        link.style.display = 'inline-block';
        link.style.whiteSpace = 'nowrap';
      });
      
      // Apply inline display to the container
      container.style.display = 'inline-flex';
      container.style.flexWrap = 'nowrap';
      container.style.whiteSpace = 'nowrap';
      container.style.overflow = 'visible';
      
      // Find parent span with mr1 nowrap class
      let parent = container.closest('.mr1.nowrap');
      if (parent) {
        parent.style.display = 'flex';
        parent.style.flexWrap = 'nowrap';
        parent.style.alignItems = 'center';
        parent.style.whiteSpace = 'nowrap';
        parent.style.overflow = 'visible';
      }
    });
    
    // Fix for possible unexpected line breaks
    document.querySelectorAll('.tag-separator').forEach(sep => {
      sep.style.display = 'inline-block';
      sep.style.whiteSpace = 'nowrap';
    });
  });
})();
