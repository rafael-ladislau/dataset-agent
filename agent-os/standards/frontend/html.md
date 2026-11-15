## HTML Structure and Formatting

### Structure Rules
- Use 2 spaces for indentation
- Place nested elements on new lines with proper indentation
- Content between tags should be on its own line when multi-line

### Attribute Formatting
- Place each HTML attribute on its own line
- Align attributes vertically
- Keep the closing `>` on the same line as the last attribute

### Example HTML Structure

```html
<div class="container">
  <header class="flex flex-col space-y-2
                 md:flex-row md:space-y-0 md:space-x-4">
    <h1 class="text-primary dark:text-primary-300">
      Page Title
    </h1>
    <nav class="flex flex-col space-y-2
                md:flex-row md:space-y-0 md:space-x-4">
      <a href="/"
         class="btn-ghost">
        Home
      </a>
      <a href="/about"
         class="btn-ghost">
        About
      </a>
    </nav>
  </header>
</div>
```

## Semantic HTML Testing Patterns

### Accessibility-First Testing
- Use semantic HTML elements as testing contracts
- Test accessibility features, not just visual appearance
- Include ARIA labels and roles for complex interactions

```html
<!-- ✅ Good: Semantic and testable -->
<form class="user-registration-form" 
      data-testid="registration-form"
      role="form"
      aria-labelledby="registration-title">
  
  <h2 id="registration-title">Create Account</h2>
  
  <fieldset class="user-info">
    <legend>Personal Information</legend>
    
    <div class="form-group">
      <label for="email" class="required">
        Email Address
      </label>
      <input type="email" 
             id="email" 
             name="email"
             data-testid="email-input"
             aria-required="true"
             aria-describedby="email-error">
      <div id="email-error" 
           class="error-message" 
           data-testid="email-error"
           role="alert"
           aria-live="polite">
      </div>
    </div>
    
    <div class="form-group">
      <label for="password" class="required">
        Password
      </label>
      <input type="password" 
             id="password" 
             name="password"
             data-testid="password-input"
             aria-required="true"
             aria-describedby="password-help password-error">
      <div id="password-help" class="help-text">
        Must be at least 8 characters
      </div>
      <div id="password-error" 
           class="error-message" 
           data-testid="password-error"
           role="alert"
           aria-live="polite">
      </div>
    </div>
  </fieldset>
  
  <div class="form-actions">
    <button type="submit" 
            class="btn-primary"
            data-testid="submit-button"
            aria-describedby="submit-status">
      Create Account
    </button>
    <div id="submit-status" 
         data-testid="submit-status"
         role="status"
         aria-live="polite">
    </div>
  </div>
</form>
```

### Component Interface Testing
- Define clear component boundaries with data-testid
- Use consistent naming patterns for test selectors
- Include state indicators for dynamic content

```html
<!-- ✅ Good: Clear component interfaces -->
<article class="blog-post" 
         data-testid="blog-post"
         data-post-id="123">
  
  <header class="post-header" data-testid="post-header">
    <h1 class="post-title" data-testid="post-title">
      Building Modular Applications
    </h1>
    <div class="post-meta" data-testid="post-meta">
      <time datetime="2023-12-01" data-testid="post-date">
        December 1, 2023
      </time>
      <address class="post-author" data-testid="post-author">
        <a href="/author/john-doe" rel="author">John Doe</a>
      </address>
    </div>
  </header>
  
  <div class="post-content" data-testid="post-content">
    <p>Content goes here...</p>
  </div>
  
  <footer class="post-footer" data-testid="post-footer">
    <div class="post-tags" data-testid="post-tags">
      <span class="tag" data-testid="tag" data-tag="architecture">Architecture</span>
      <span class="tag" data-testid="tag" data-tag="development">Development</span>
    </div>
    
    <div class="post-actions" data-testid="post-actions">
      <button class="like-button" 
              data-testid="like-button"
              data-post-id="123"
              aria-pressed="false">
        <span class="like-count" data-testid="like-count">42</span>
        Like
      </button>
      
      <button class="share-button" 
              data-testid="share-button"
              data-post-id="123">
        Share
      </button>
    </div>
  </footer>
</article>
```

### Navigation Testing Patterns
- Use semantic navigation elements
- Include skip links for accessibility
- Test navigation state and active indicators

```html
<!-- ✅ Good: Testable navigation -->
<nav class="main-navigation" 
     data-testid="main-nav"
     role="navigation"
     aria-label="Main navigation">
  
  <a href="#main-content" 
     class="skip-link"
     data-testid="skip-link">
    Skip to main content
  </a>
  
  <div class="nav-brand" data-testid="nav-brand">
    <a href="/" 
       data-testid="home-link"
       aria-label="Home">
      <img src="logo.svg" alt="Company Logo">
    </a>
  </div>
  
  <ul class="nav-menu" 
      data-testid="nav-menu"
      role="menubar">
    <li class="nav-item" role="none">
      <a href="/products" 
         class="nav-link"
         data-testid="products-link"
         role="menuitem"
         aria-current="page">
        Products
      </a>
    </li>
    <li class="nav-item" role="none">
      <a href="/about" 
         class="nav-link"
         data-testid="about-link"
         role="menuitem">
        About
      </a>
    </li>
    <li class="nav-item" role="none">
      <a href="/contact" 
         class="nav-link"
         data-testid="contact-link"
         role="menuitem">
        Contact
      </a>
    </li>
  </ul>
  
  <div class="nav-actions" data-testid="nav-actions">
    <button class="mobile-menu-toggle" 
            data-testid="mobile-menu-toggle"
            aria-expanded="false"
            aria-controls="nav-menu"
            aria-label="Toggle navigation menu">
      <span class="hamburger-icon" aria-hidden="true"></span>
    </button>
  </div>
</nav>
```

### Data Table Testing
- Use proper table semantics for screen readers
- Include sortable column indicators
- Test pagination and filtering interfaces

```html
<!-- ✅ Good: Testable data table -->
<div class="table-container" data-testid="users-table-container">
  
  <div class="table-controls" data-testid="table-controls">
    <div class="table-search" data-testid="table-search">
      <label for="user-search" class="sr-only">Search users</label>
      <input type="search" 
             id="user-search"
             data-testid="search-input"
             placeholder="Search users..."
             aria-describedby="search-help">
      <div id="search-help" class="sr-only">
        Search by name, email, or role
      </div>
    </div>
    
    <div class="table-filters" data-testid="table-filters">
      <select data-testid="role-filter" aria-label="Filter by role">
        <option value="">All Roles</option>
        <option value="admin">Admin</option>
        <option value="user">User</option>
      </select>
    </div>
  </div>
  
  <table class="data-table" 
         data-testid="users-table"
         role="table"
         aria-label="Users list">
    <caption class="sr-only">
      List of users with their roles and status
    </caption>
    
    <thead data-testid="table-header">
      <tr role="row">
        <th scope="col" 
            data-testid="name-header"
            aria-sort="ascending">
          <button class="sort-button" 
                  data-testid="sort-name"
                  data-sort="name"
                  aria-label="Sort by name, currently ascending">
            Name
            <span class="sort-icon" aria-hidden="true">↑</span>
          </button>
        </th>
        <th scope="col" 
            data-testid="email-header">
          Email
        </th>
        <th scope="col" 
            data-testid="role-header">
          Role
        </th>
        <th scope="col" 
            data-testid="status-header">
          Status
        </th>
        <th scope="col" 
            data-testid="actions-header">
          <span class="sr-only">Actions</span>
        </th>
      </tr>
    </thead>
    
    <tbody data-testid="table-body">
      <tr data-testid="user-row" data-user-id="1" role="row">
        <td data-testid="user-name">John Doe</td>
        <td data-testid="user-email">john@example.com</td>
        <td data-testid="user-role">
          <span class="role-badge role-admin" data-testid="role-badge">
            Admin
          </span>
        </td>
        <td data-testid="user-status">
          <span class="status-indicator status-active" 
                data-testid="status-indicator"
                aria-label="Active user">
            Active
          </span>
        </td>
        <td data-testid="user-actions">
          <button class="btn-edit" 
                  data-testid="edit-user"
                  data-user-id="1"
                  aria-label="Edit John Doe">
            Edit
          </button>
          <button class="btn-delete" 
                  data-testid="delete-user"
                  data-user-id="1"
                  aria-label="Delete John Doe">
            Delete
          </button>
        </td>
      </tr>
    </tbody>
  </table>
  
  <div class="table-pagination" data-testid="table-pagination">
    <div class="pagination-info" data-testid="pagination-info">
      Showing 1-10 of 50 users
    </div>
    <nav class="pagination-nav" 
         data-testid="pagination-nav"
         aria-label="Table pagination">
      <button class="pagination-prev" 
              data-testid="prev-page"
              disabled
              aria-label="Previous page">
        Previous
      </button>
      <button class="pagination-next" 
              data-testid="next-page"
              aria-label="Next page">
        Next
      </button>
    </nav>
  </div>
</div>
```

### HTML Testing Anti-Patterns

```html
<!-- ❌ Bad: Non-semantic, hard to test -->
<div class="button" onclick="submitForm()">Submit</div>
<div class="table">
  <div class="row">
    <div class="cell">Name</div>
    <div class="cell">Email</div>
  </div>
</div>

<!-- ✅ Good: Semantic, accessible, testable -->
<button type="submit" 
        data-testid="submit-button"
        onclick="submitForm()">
  Submit
</button>
<table data-testid="users-table">
  <thead>
    <tr>
      <th scope="col" data-testid="name-header">Name</th>
      <th scope="col" data-testid="email-header">Email</th>
    </tr>
  </thead>
</table>

<!-- ❌ Bad: Generic test selectors -->
<div class="modal">
  <div class="content">
    <h2>Delete User</h2>
    <p>Are you sure?</p>
    <button class="btn">Yes</button>
    <button class="btn">No</button>
  </div>
</div>

<!-- ✅ Good: Specific, meaningful test selectors -->
<div class="modal" 
     data-testid="delete-user-modal"
     role="dialog"
     aria-labelledby="modal-title"
     aria-describedby="modal-description">
  <div class="modal-content">
    <h2 id="modal-title" data-testid="modal-title">Delete User</h2>
    <p id="modal-description" data-testid="modal-description">
      Are you sure you want to delete this user?
    </p>
    <div class="modal-actions">
      <button class="btn-danger" 
              data-testid="confirm-delete"
              data-action="delete">
        Yes, Delete
      </button>
      <button class="btn-secondary" 
              data-testid="cancel-delete"
              data-action="cancel">
        Cancel
      </button>
    </div>
  </div>
</div>
```

