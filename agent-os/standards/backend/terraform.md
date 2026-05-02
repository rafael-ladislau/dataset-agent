## Infrastructure as Code (IaC) Best Practices

### Terraform Modular Architecture

Apply the same modular principles to infrastructure code:

#### Module Structure
```
infrastructure/
├── modules/
│   ├── networking/
│   │   ├── main.tf           # Core networking resources
│   │   ├── variables.tf      # Input parameters
│   │   ├── outputs.tf        # Exposed values
│   │   └── README.md         # Usage documentation
│   ├── compute/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   └── storage/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
└── shared/
    ├── data.tf              # Shared data sources
    └── locals.tf            # Common local values
```

#### Darwin Seguros Module Integration
```hcl
# ✅ Good: Use company modules with version pinning
module "notification_queue" {
  source  = "git::https://github.com/darwin-seguros/terraform-module-sqs.git?ref=v1.2.0"
  
  queue_name = "user-notifications-${var.environment}"
  
  # Module-specific configuration
  visibility_timeout_seconds = 300
  message_retention_seconds  = 1209600
  
  tags = local.common_tags
}

module "user_topic" {
  source = "git::https://github.com/darwin-seguros/terraform-module-sns.git?ref=v1.1.0"
  
  topic_name = "user-events-${var.environment}"
  
  tags = local.common_tags
}

# Subscribe SQS to SNS using company module
module "queue_subscription" {
  source = "git::https://github.com/darwin-seguros/terraform-module-sns-subscription.git?ref=v1.0.0"
  
  topic_arn = module.user_topic.topic_arn
  endpoint  = module.notification_queue.queue_arn
  protocol  = "sqs"
}
```

#### Interface-First Infrastructure Design
```hcl
# ✅ Good: Define clear module interfaces
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "application_name" {
  description = "Name of the application"
  type        = string
  validation {
    condition     = length(var.application_name) > 0
    error_message = "Application name cannot be empty."
  }
}

# Clear outputs for other modules
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

output "database_endpoint" {
  description = "RDS cluster endpoint"
  value       = aws_rds_cluster.main.endpoint
  sensitive   = true
}
```

### Terraform Security and Error Handling

#### Sensitive Data Management
```hcl
# ✅ Good: Use AWS Secrets Manager for sensitive data
data "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = "rds-credentials-${var.environment}"
}

locals {
  db_creds = jsondecode(data.aws_secretsmanager_secret_version.db_credentials.secret_string)
}

resource "aws_db_instance" "main" {
  # Use secrets instead of hardcoded values
  username = local.db_creds.username
  password = local.db_creds.password
  
  # Other configuration...
}

# ❌ Bad: Hardcoded sensitive values
resource "aws_db_instance" "bad" {
  username = "admin"           # Hardcoded username
  password = "mypassword123"   # Hardcoded password - NEVER DO THIS
}
```

#### Resource Dependencies and Error Handling
```hcl
# ✅ Good: Explicit dependencies and error handling
resource "aws_security_group" "app" {
  name_prefix = "${var.application_name}-app-"
  vpc_id      = var.vpc_id

  # Explicit dependency management
  depends_on = [aws_vpc.main]

  lifecycle {
    create_before_destroy = true
  }

  tags = merge(local.common_tags, {
    Name = "${var.application_name}-app-sg"
  })
}

resource "aws_instance" "app" {
  count = var.instance_count

  ami           = data.aws_ami.app.id
  instance_type = var.instance_type
  
  # Handle conditional configurations
  vpc_security_group_ids = var.vpc_id != null ? [aws_security_group.app.id] : []
  
  # Error handling with validation
  user_data = templatefile("${path.module}/user_data.sh", {
    app_name    = var.application_name
    environment = var.environment
  })

  tags = merge(local.common_tags, {
    Name = "${var.application_name}-app-${count.index + 1}"
  })
}
```

### Terraform Testing Patterns

#### Module Testing with Terratest
```go
// ✅ Good: Test infrastructure modules like code modules
func TestVPCModule(t *testing.T) {
    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/networking",
        Vars: map[string]interface{}{
            "environment":      "test",
            "application_name": "test-app",
            "cidr_block":      "10.0.0.0/16",
        },
    }

    defer terraform.Destroy(t, terraformOptions)
    terraform.InitAndApply(t, terraformOptions)

    // Test outputs
    vpcId := terraform.Output(t, terraformOptions, "vpc_id")
    assert.NotEmpty(t, vpcId)

    // Test actual AWS resources
    vpc := aws.GetVpcById(t, vpcId, "us-east-1")
    assert.Equal(t, "10.0.0.0/16", vpc.CidrBlock)
}
```

#### Integration Testing
```hcl
# ✅ Good: Test complete infrastructure stacks
module "test_environment" {
  source = "../environments/dev"
  
  environment      = "test"
  application_name = "integration-test"
  
  # Override for testing
  instance_count = 1
  instance_type  = "t3.micro"
}

# Test connectivity and functionality
resource "null_resource" "integration_test" {
  depends_on = [module.test_environment]
  
  provisioner "local-exec" {
    command = "./scripts/run_integration_tests.sh ${module.test_environment.app_endpoint}"
  }
}
```

### Terraform Anti-Patterns to Avoid

#### Avoid Tight Coupling
```hcl
# ❌ Bad: Tight coupling between resources
resource "aws_instance" "web" {
  # Hardcoded references make it hard to replace
  subnet_id         = aws_subnet.web.id
  security_group_ids = [aws_security_group.web.id]
  
  # Embedded configuration makes testing difficult
  user_data = <<-EOF
    #!/bin/bash
    echo "server_name ${aws_db_instance.main.endpoint}" > /etc/myapp.conf
    systemctl start myapp
  EOF
}
```

```hcl
# ✅ Good: Modular, replaceable design
module "compute" {
  source = "./modules/compute"
  
  # Pass dependencies as parameters
  subnet_ids         = module.networking.private_subnet_ids
  security_group_ids = [module.security.app_security_group_id]
  
  # Configuration through variables
  app_config = {
    database_endpoint = module.database.endpoint
    redis_endpoint   = module.cache.endpoint
  }
  
  environment      = var.environment
  application_name = var.application_name
}
```

#### Avoid Resource Sprawl
```hcl
# ❌ Bad: All resources in one file
resource "aws_vpc" "main" { }
resource "aws_subnet" "public" { }
resource "aws_subnet" "private" { }
resource "aws_internet_gateway" "main" { }
resource "aws_nat_gateway" "main" { }
resource "aws_route_table" "public" { }
resource "aws_route_table" "private" { }
resource "aws_security_group" "web" { }
resource "aws_security_group" "app" { }
resource "aws_security_group" "db" { }
resource "aws_instance" "web" { }
resource "aws_instance" "app" { }
resource "aws_db_instance" "main" { }
# ... 50+ more resources
```

```hcl
# ✅ Good: Organized into logical modules
module "networking" {
  source = "./modules/networking"
  # networking-related resources
}

module "security" {
  source = "./modules/security"
  vpc_id = module.networking.vpc_id
}

module "compute" {
  source = "./modules/compute"
  subnet_ids = module.networking.private_subnet_ids
}

module "database" {
  source = "./modules/database"
  subnet_ids = module.networking.database_subnet_ids
}
```

## Terraform Code Style Guide

### General Formatting

**Indentation and Spacing:**
- Use 2 spaces for indentation (never tabs)
- Add blank lines between resource blocks for readability
- Align multi-line argument lists for complex resources
- Use trailing commas in lists and maps when spanning multiple lines

**File Organization:**
- **main.tf**: Primary resource definitions
- **variables.tf**: Input variable declarations
- **outputs.tf**: Output value declarations
- **locals.tf**: Local value definitions (optional)
- **data.tf**: Data source declarations (optional)
- **versions.tf**: Provider version constraints

**Naming Conventions:**
- **Resources**: Use snake_case with descriptive names (e.g., `user_notification_queue`)
- **Variables**: Use snake_case with clear, descriptive names (e.g., `database_instance_class`)
- **Outputs**: Use snake_case describing what is being output (e.g., `vpc_id`, `database_endpoint`)
- **Locals**: Use snake_case for consistency (e.g., `common_tags`)
- **Files**: Use snake_case with descriptive names (e.g., `user_management.tf`)

### Resource Formatting

**Basic Resource Structure:**
```hcl
# ✅ Good: Well-formatted resource with proper spacing
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${var.application_name}-vpc"
  })
}

# ❌ Bad: Poor formatting and spacing
resource "aws_vpc" "main" {
cidr_block=var.vpc_cidr
enable_dns_hostnames=true
enable_dns_support=true
tags={
Name="${var.application_name}-vpc"
Environment=var.environment
}
}
```

### Variable Declarations

**Variable Structure:**
```hcl
# ✅ Good: Complete variable declaration with validation
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition = contains([
      "dev",
      "staging", 
      "prod",
    ], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "database_config" {
  description = "Database configuration parameters"
  type = object({
    instance_class    = string
    allocated_storage = number
    backup_retention  = number
    multi_az         = bool
  })
  
  default = {
    instance_class    = "db.t3.micro"
    allocated_storage = 20
    backup_retention  = 7
    multi_az         = false
  }
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the application"
  type        = list(string)
  default     = []
  
  validation {
    condition     = length(var.allowed_cidr_blocks) > 0
    error_message = "At least one CIDR block must be specified."
  }
}
```

### Local Values

**Locals Structure:**
```hcl
# ✅ Good: Organized local values with clear grouping
locals {
  # Common resource tags
  common_tags = {
    Environment     = var.environment
    Application     = var.application_name
    ManagedBy      = "terraform"
    CostCenter     = var.cost_center
    Owner          = var.owner_email
    CreatedDate    = formatdate("YYYY-MM-DD", timestamp())
  }

  # Network configuration
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
  
  private_subnet_cidrs = [
    for i, az in local.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i + 10)
  ]
  
  public_subnet_cidrs = [
    for i, az in local.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i + 20)
  ]

  # Application configuration
  app_config = {
    port         = var.app_port
    health_check = "/health"
    log_level    = var.environment == "prod" ? "info" : "debug"
  }
}
```

### Comments and Documentation

**Resource Comments:**
```hcl
# Create VPC with DNS resolution enabled for EKS compatibility
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true  # Required for EKS
  enable_dns_support   = true  # Required for EKS

  tags = merge(local.common_tags, {
    Name = "${var.application_name}-vpc"
    # EKS cluster discovery tag
    "kubernetes.io/cluster/${var.application_name}-${var.environment}" = "shared"
  })
}

# Security group for application servers
# Allows inbound HTTP/HTTPS from load balancer only
resource "aws_security_group" "app" {
  name_prefix = "${var.application_name}-app-"
  description = "Security group for ${var.application_name} application servers"
  vpc_id      = aws_vpc.main.id

  # HTTP from load balancer
  ingress {
    description     = "HTTP from ALB"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  tags = merge(local.common_tags, {
    Name = "${var.application_name}-app-sg"
  })
}
```

### String Formatting

**String Interpolation:**
```hcl
# ✅ Good: Clear string interpolation with proper spacing
locals {
  bucket_name = "${var.application_name}-${var.environment}-assets"
  
  database_name = join("-", [
    var.application_name,
    var.environment,
    "db"
  ])
  
  # Multi-line string with proper indentation
  user_data = templatefile("${path.module}/templates/user_data.sh", {
    app_name    = var.application_name
    environment = var.environment
    log_level   = local.app_config.log_level
  })
}

# ❌ Bad: Unclear concatenation and poor formatting
locals {
  bucket_name="${var.application_name}${var.environment}assets"
  database_name=var.application_name+"-"+var.environment+"-db"
}
```

### Conditional Expressions
```hcl
# ✅ Good: Readable conditional expressions
resource "aws_instance" "app" {
  count = var.enable_app_server ? var.instance_count : 0

  instance_type = var.environment == "prod" ? "t3.large" : "t3.micro"
  
  vpc_security_group_ids = var.vpc_id != null ? [
    aws_security_group.app.id,
    data.aws_security_group.common.id,
  ] : []

  tags = merge(local.common_tags, {
    Name = "${var.application_name}-app-${count.index + 1}"
    Type = var.environment == "prod" ? "production" : "development"
  })
}
```

### Version Constraints

**Provider Versioning:**
```hcl
# ✅ Good: Specific version constraints
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # Allow patch updates, lock minor version
    }
    
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# ❌ Bad: No version constraints (can cause breaking changes)
terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}
```

### Workflow for Infrastructure Changes

1. **Design the module interface** - what resources should be exposed?
2. **Write Terratest tests** - define expected infrastructure behavior
3. **Implement Terraform modules** - hide AWS complexity behind clean interfaces
4. **Test with `terraform plan`** - verify changes before applying
5. **Document module usage** - provide clear examples and parameter descriptions

