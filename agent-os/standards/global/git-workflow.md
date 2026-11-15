## Git Flow

### Branch Structure

**Main Branches:**
- `main`: Production branch - code must be production-ready
- `develop`: Development branch - continuous development integration

**Support Branches:**
- `feature/`: New functionality development
- `bugfix/`: Bug fixes during development
- `release/`: Prepare new production versions
- `hotfix/`: Critical fixes for production

### Branch Naming Conventions

**Feature Branches:**
- Pattern: `feature/description-of-feature`
- Example: `feature/adicionar-autenticacao`
- Created from: `develop`
- Merged into: `develop`

**Bug Fix Branches:**
- Pattern: `bugfix/description-of-bug`
- Example: `bugfix/corrigir-erro-login`
- Created from: `develop`
- Merged into: `develop`

**Release Branches:**
- Pattern: `release/MAJOR.MINOR.PATCH-rc.RELEASE_VERSION`
- Example: `release/1.1.0-rc.2`
- Created from: `develop`
- Merged into: `main` and `develop`

**Hotfix Branches:**
- Pattern: `hotfix/version-description-of-bug`
- Example: `hotfix/1.0.1-corrigir-npe`
- Created from: `main`
- Merged into: `main` and `develop`

### Semantic Versioning

Follow SemVer (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible new features
- **PATCH**: Backward-compatible bug fixes

**Examples:**
- `1.0.0`: First stable version
- `1.1.0`: New features, backward-compatible
- `1.1.1`: Bug fixes for version 1.1.0

### Git Flow Process

**1. Feature Development:**
```bash
# Ensure you're on develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/nome-da-feature develop

# After development, create PR to develop
# Ensure all tests pass before merge
```

**2. Bug Fix Development:**
```bash
# Ensure you're on develop
git checkout develop
git pull origin develop

# Create bugfix branch
git checkout -b bugfix/description-of-bug develop

# After fix, create PR to develop
```

**3. Release Preparation:**
```bash
# Ensure you're on develop
git checkout develop
git pull origin develop

# Create release branch
git checkout -b release/1.1.0-rc.1 develop

# Perform final adjustments and bug fixes
# Create PR to main for deployment
```

**4. Hotfix for Production:**
```bash
# Ensure you're on main
git checkout main
git pull origin main

# Create hotfix branch
git checkout -b hotfix/1.0.1-corrigir-bug main

# Fix critical bug
# Merge into both main and develop
```

### Important Notes

**Branch Synchronization:**
- If `develop` is outdated compared to `main`:
  - Infrastructure team creates `develop-old` from current `develop`
  - New `develop` is created from `main`
  - Continue development from new `develop`

**Pull Request Requirements:**
- All tests must pass before merge
- Code review required for feature/bugfix PRs
- Release PRs require deployment coordination

**Deployment Process:**
- After release branch PR to main: create deployment ticket
- Schedule release with infrastructure team
- Follow deployment procedures for production releases

### Commit Message Conventions

Use clear, descriptive commit messages:
- Start with a verb in present tense
- Keep the first line under 50 characters
- Add detailed description after blank line if needed

**Examples:**
```
Add user authentication feature
Fix memory leak in payment processor
Update database migration scripts
```

**Conventional Commits (optional but recommended):**
```
feat: add user authentication
fix: resolve memory leak in payment processor
docs: update API documentation
refactor: simplify order processing logic
test: add integration tests for checkout
chore: update dependencies
```

