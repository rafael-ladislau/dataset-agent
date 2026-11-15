## GitHub Actions & CI/CD Patterns

### Exit Code Handling

When working with external tools (terraform, docker, etc.) in GitHub Actions:

✅ **DO: Capture and handle exit codes explicitly**
```bash
set +e  # Don't fail immediately
terraform plan -out=plan.tfplan
PLAN_EXIT_CODE=$?
set -e  # Re-enable exit on error

case $PLAN_EXIT_CODE in
  0) echo "✅ No changes needed" ;;
  1) echo "📋 Changes detected" ;;
  2) echo "❌ Error occurred"; exit 1 ;;
esac
```

❌ **DON'T: Let tools fail silently or misinterpret exit codes**
```bash
# This will fail when terraform detects changes (exit code 1)
terraform plan -out=plan.tfplan
```

### Environment-Specific Configuration

Use environment detection with fallback patterns:

✅ **DO: Environment-aware configuration with file validation**
```bash
case "$ENVIRONMENT" in
  "dev"|"development") CONFIG_FILE="development.config" ;;
  "staging") CONFIG_FILE="staging.config" ;;
  "prod"|"production") CONFIG_FILE="production.config" ;;
  *) CONFIG_FILE="development.config" ;;  # Fallback
esac

if [ -f "$CONFIG_FILE" ]; then
  echo "✅ Using config: $CONFIG_FILE"
  use_config "$CONFIG_FILE"
else
  echo "⚠️ Config not found: $CONFIG_FILE"
  list_available_configs
  fallback_action
fi
```

❌ **DON'T: Assume files exist or use hardcoded environments**
```bash
# This will fail if file doesn't exist
terraform init -backend-config="production.tfbackend"
```

### Logging and Observability

Use structured logging with clear status indicators:

✅ **DO: Structured logging with GitHub Actions groups and emojis**
```bash
echo "::group::🔧 Infrastructure Setup"
echo "🌍 Environment: $ENVIRONMENT"
echo "📋 Using configuration: $CONFIG_FILE"
echo "✅ Setup completed successfully"
echo "::endgroup::"
```

❌ **DON'T: Use plain text without structure or context**
```bash
echo "Setting up infrastructure"
echo "Done"
```

### Error Recovery and Debugging

Provide actionable error messages and debugging information:

✅ **DO: Helpful error messages with debugging context**
```bash
if [ ! -f "$REQUIRED_FILE" ]; then
  echo "❌ Required file not found: $REQUIRED_FILE"
  echo "📋 Expected files for $ENVIRONMENT environment:"
  echo "  - development.config (for dev)"
  echo "  - staging.config (for staging)"  
  echo "  - production.config (for prod)"
  echo "📁 Available files:"
  ls -la *.config 2>/dev/null || echo "No config files found"
  echo "💡 Create the appropriate config file for your environment"
  exit 1
fi
```

❌ **DON'T: Generic error messages without context**
```bash
if [ ! -f "$REQUIRED_FILE" ]; then
  echo "File not found"
  exit 1
fi
```

### Workflow Dependencies and Job Ordering

Structure jobs with clear dependencies and conditional execution:

✅ **DO: Explicit job dependencies with conditional execution**
```yaml
jobs:
  setup:
    # Foundation job
  
  generate-artifacts:
    needs: [setup]
    if: ${{ !cancelled() && inputs.skip-artifacts != true }}
    
  validate:
    needs: [setup, generate-artifacts]  # Explicit dependency
    if: ${{ !cancelled() && needs.setup.outputs.environment != 'local' }}
    
  deploy:
    needs: [validate]
    if: ${{ !cancelled() && needs.validate.result == 'success' }}
```

❌ **DON'T: Implicit dependencies or race conditions**
```yaml
jobs:
  validate:
    # Missing dependency on artifact generation
    
  deploy:
    # Missing proper conditionals
```

### Composite Action Design Patterns

Create modular, reusable composite actions with clear interfaces:

✅ **DO: Modular composite actions with proper input validation**
```yaml
# composite action: setup-stack/action.yml
name: 'Setup Development Stack'
description: 'Configure Node.js or .NET development environment'
inputs:
  stack-type:
    description: 'Stack type (node, dotnet)'
    required: true
  npm-access-token:
    description: 'NPM access token for private packages'
    required: false

runs:
  using: 'composite'
  steps:
    - name: Validate Inputs
      shell: bash
      run: |
        if [ -z "${{ inputs.stack-type }}" ]; then
          echo "❌ stack-type is required"
          exit 1
        fi
        
        case "${{ inputs.stack-type }}" in
          "node"|"dotnet") echo "✅ Valid stack type: ${{ inputs.stack-type }}" ;;
          *) echo "❌ Invalid stack type: ${{ inputs.stack-type }}"; exit 1 ;;
        esac
```

❌ **DON'T: Monolithic actions without input validation**
```yaml
# Single action trying to do everything without validation
runs:
  using: 'composite'
  steps:
    - name: Do Everything
      shell: bash
      run: |
        # Hundreds of lines doing multiple unrelated things
        # No input validation or error handling
```

### Migration Detection and Approval Patterns

Implement accurate migration detection with proper approval workflows:

✅ **DO: Accurate migration detection with simplified approval**
```yaml
# Use the step that actually checks for pending migrations
- name: Check for Pending Migrations
  id: check-migrations
  run: |
    if npm run typeorm migration:show | grep -q "\\[ \\]"; then
      echo "has-migrations=true" >> $GITHUB_OUTPUT
    else
      echo "has-migrations=false" >> $GITHUB_OUTPUT
    fi

# Reference the correct step
apply-migrations:
  needs: [validate-migrations]
  environment: ${{ inputs.require-approval && format('{0}-approval', inputs.environment) || inputs.environment }}
  if: ${{ needs.validate-migrations.outputs.has-migrations == 'true' }}
```

❌ **DON'T: False positive detection or redundant approval jobs**
```yaml
# Wrong step reference causing false positives
apply-migrations:
  if: ${{ needs.validate-migrations.outputs.wrong-step-output == 'true' }}

# Unnecessary separate approval job
approve-migrations:
  needs: [validate-migrations]
  environment: approval
  # Redundant job that could be integrated into apply-migrations
```

### External Repository Compatibility

Design workflows and actions for external repository usage:

✅ **DO: Full repository paths and simplified action variants**
```yaml
# In reusable workflow - use full paths
- name: Setup Environment
  uses: darwin-seguros/reusable-actions/.github/actions/setup-stack@feature/refactoring

# Create simplified variants for external use
# setup-stack-simple/action.yml - uses only pre-installed tools
# security-scan-simple/action.yml - avoids complex installations
```

❌ **DON'T: Relative paths or complex tool dependencies**
```yaml
# This fails when called from external repositories
- uses: ./.github/actions/setup-stack

# Complex action requiring tool installation on every run
- name: Complex Security Scan
  uses: ./.github/actions/security-scan  # Installs gitleaks, trivy, etc.
```

### Runner Environment Management

Handle different runner environments appropriately:

✅ **DO: Environment-aware tool installation and VPN routing**
```yaml
# For VPN-dependent operations
gitops-deploy:
  runs-on:
    group: darwin-shared  # Has VPN access

  steps:
    - name: Install Required Tools
      shell: bash
      run: |
        # Check and install AWS CLI if missing
        if ! command -v aws >/dev/null 2>&1; then
          curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
          unzip -q awscliv2.zip && sudo ./aws/install && rm -rf awscliv2.zip aws/
        fi
        
        # Check and install jq if missing  
        if ! command -v jq >/dev/null 2>&1; then
          sudo apt-get update -qq && sudo apt-get install -y jq
        fi
```

❌ **DON'T: Assume tool availability or ignore VPN requirements**
```yaml
# This fails on custom runners
deploy:
  runs-on: ubuntu-latest  # No VPN for ArgoCD
  steps:
    - run: aws sts get-caller-identity  # Assumes AWS CLI exists
    - run: argocd login $SERVER  # Can't reach internal ArgoCD
```

## YAML & GitHub Actions Style Guide

### YAML Formatting

**Indentation:**
- Use 2 spaces for indentation (never tabs)
- Maintain consistent indentation levels
- Align list items properly

✅ **DO:**
```yaml
jobs:
  setup:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
```

❌ **DON'T:**
```yaml
jobs:
setup:
  runs-on: ubuntu-latest
  steps:
  - name: Checkout
    uses: actions/checkout@v4
```

**String Quoting:**
- Use single quotes for simple strings
- Use double quotes when interpolation is needed
- Quote strings that contain special characters

✅ **DO:**
```yaml
environment: 'production'
message: "Environment: ${{ needs.setup.outputs.environment }}"
version: '1.0.0'
```

❌ **DON'T:**
```yaml
environment: production  # Unquoted
message: 'Environment: ${{ needs.setup.outputs.environment }}'  # Wrong quotes for interpolation
```

### Job and Step Naming

- Use descriptive, action-oriented names
- Include emojis for visual clarity in logs
- Use consistent emoji patterns across workflows

✅ **DO:**
```yaml
jobs:
  setup:
    name: 🚀 Environment Setup
    
  validate:
    name: ✅ Terraform Validation
    steps:
      - name: 🔧 Terraform Init
      - name: 📋 Terraform Plan
      - name: ✅ Validate Configuration
```

❌ **DON'T:**
```yaml
jobs:
  job1:
    name: Setup
    
  job2:
    name: Validation
    steps:
      - name: Init
      - name: Plan
```

### Conditional Logic

- Use explicit conditionals with clear logic
- Include cancellation checks for dependent jobs
- Group related conditions logically

✅ **DO:**
```yaml
if: |
  !cancelled() &&
  needs.setup.outputs.environment != 'local' &&
  (
    github.event_name == 'push' ||
    github.event_name == 'workflow_dispatch'
  )
```

❌ **DON'T:**
```yaml
if: needs.setup.outputs.environment != 'local' && github.event_name == 'push' || github.event_name == 'workflow_dispatch'
```

### Reusable Workflows & Composite Actions

**Composite Action Path References:**
- Use full repository paths for external composite actions
- Never use relative paths in reusable workflows
- Always include branch/tag reference

✅ **DO:**
```yaml
# In reusable workflow calling composite action
- uses: darwin-seguros/reusable-actions/.github/actions/setup-aws@feature/refactoring

# In main workflow calling composite action
- uses: ./.github/actions/setup-stack@main  # Local only
```

❌ **DON'T:**
```yaml
# This fails when called from external repositories
- uses: ./.github/actions/setup-aws
```

**Reusable Workflow Interfaces:**
- Define clear input and secret interfaces
- Use `secrets: inherit` when passing all secrets
- Avoid passing individual secrets when possible

✅ **DO:**
```yaml
# Calling reusable workflow
uses: ./.github/workflows/reusable-setup.yml
with:
  environment: ${{ needs.setup.outputs.environment }}
  stack-type: 'node'
secrets: inherit

# Reusable workflow definition
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      stack-type:
        required: false
        type: string
        default: 'node'
```

**Context Access Limitations:**
- `vars` and `secrets` contexts are NOT available in composite actions
- Pass variables as inputs from workflows to composite actions
- Use environment-specific logic in workflows, not composite actions

✅ **DO:**
```yaml
# In workflow file
- name: Deploy to Environment
  uses: ./.github/actions/deploy@main
  with:
    argocd-server: ${{ vars.ARGOCD_SERVER_PROD || vars.ARGOCD_SERVER }}
    argocd-token: ${{ secrets.ARGOCD_TOKEN_PROD || secrets.ARGOCD_TOKEN }}

# In composite action
inputs:
  argocd-server:
    description: 'ArgoCD server URL'
    required: true
  argocd-token:
    description: 'ArgoCD token'
    required: true
```

### Action Versions

- Pin action versions to specific commits or tags
- Use latest stable versions
- Document version update policies

✅ **DO:**
```yaml
- uses: actions/checkout@v4
- uses: hashicorp/setup-terraform@v3.1.2
```

❌ **DON'T:**
```yaml
- uses: actions/checkout@main
- uses: hashicorp/setup-terraform@latest
```

## Security and Secrets Management

### NPM Authentication

- Create `.npmrc` file for GitHub Packages access
- Use organization-level NPM tokens
- Configure before Docker builds that need private packages

✅ **DO:**
```yaml
- name: Configure NPM for GitHub Packages
  run: |
    echo "@darwin-seguros:registry=https://npm.pkg.github.com" >> .npmrc
    echo "//npm.pkg.github.com/:_authToken=${{ secrets.NPM_ACCESS_TOKEN }}" >> .npmrc

- name: Build Docker Image
  run: docker build -t $IMAGE_URI .
```

### Secret Validation

- Always validate required secrets are available
- Provide debugging information for missing secrets
- Use environment-specific secret selection

✅ **DO:**
```yaml
- name: Validate Required Secrets
  run: |
    for SECRET in ARGOCD_SERVER ARGOCD_TOKEN; do
      if [ -z "${!SECRET}" ]; then
        echo "❌ Required secret not set: $SECRET"
        echo "🔍 Available ArgoCD vars: $(env | grep ARGOCD | cut -d= -f1)"
        exit 1
      fi
    done
```

## Debugging and Observability

### Structured Logging

- Use GitHub Actions groups for organization
- Include emojis for visual clarity
- Provide context and debugging information

✅ **DO:**
```yaml
- name: Deploy Application
  run: |
    echo "::group::🚀 Deployment Configuration"
    echo "🌍 Environment: ${{ inputs.environment }}"
    echo "📦 Service: ${{ inputs.service-name }}"
    echo "🔧 ArgoCD Server: ${{ inputs.argocd-server }}"
    echo "::endgroup::"
```

### Error Recovery

- Provide actionable error messages
- Include debugging context
- Suggest possible solutions

✅ **DO:**
```yaml
- name: Handle Deployment Failure
  if: failure()
  run: |
    echo "❌ Deployment failed"
    echo "🔍 Debug information:"
    echo "  - Environment: ${{ inputs.environment }}"
    echo "  - Service: ${{ inputs.service-name }}"
    echo "  - ArgoCD Server: ${{ inputs.argocd-server }}"
    echo "💡 Possible solutions:"
    echo "  1. Check ArgoCD server connectivity"
    echo "  2. Verify service configuration"
    echo "  3. Review previous deployment logs"
```

