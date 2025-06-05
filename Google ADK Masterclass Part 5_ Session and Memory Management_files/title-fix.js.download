/**
 * Targeted fix for blog post titles in the posts outline
 */
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    // Only run on the writing/blog page
    if (window.location.pathname.includes('/writing/') || 
        document.querySelector('.posts') || 
        document.querySelector('.post-list')) {
      
      // Target the specific structure we found in post-outline.html
      const titleHeadings = document.querySelectorAll('.col.sm-col-8 h2');
      
      titleHeadings.forEach(heading => {
        // Force the title to wrap properly
        heading.style.display = 'block';
        heading.style.wordWrap = 'break-word';
        heading.style.overflowWrap = 'break-word';
        heading.style.wordBreak = 'break-word';
        heading.style.whiteSpace = 'normal';
        heading.style.width = '100%';
        heading.style.maxWidth = '95%';
        heading.style.position = 'relative';
        heading.style.paddingRight = '0';
      });
      
      // Also target the parent containers
      const containers = document.querySelectorAll('.col.sm-col-8');
      containers.forEach(container => {
        container.style.width = '100%';
        container.style.maxWidth = '100%';
        container.style.boxSizing = 'border-box';
        container.style.paddingRight = '10px';
        container.style.float = 'none'; // Prevent floating issues
      });
      
      // Target the parent article element
      const articles = document.querySelectorAll('article.mxn2');
      articles.forEach(article => {
        // Ensure the article has sufficient width
        article.style.width = '100%';
        article.style.maxWidth = '100%';
        article.style.boxSizing = 'border-box';
        article.style.display = 'block';
      });
    }
  });
})();
