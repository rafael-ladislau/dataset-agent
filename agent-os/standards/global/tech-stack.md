## Tech Stack

Global tech stack defaults for Agent OS projects, optimized for modular architecture and black box design principles. These choices prioritize:
- **Interface-first development** - TypeScript for strong typing and contracts
- **Dependency injection** - NestJS and .NET Core support IoC containers
- **Testability** - All choices support comprehensive unit and integration testing
- **Replaceability** - Modular designs allow swapping implementations
- **Debugging ease** - Structured logging and observability built-in

Overridable in project-specific `.agent-os/product/tech-stack.md`.

### Framework & Runtime
- **Application Framework:** NestJS or .NET C#
- **Language/Runtime:** TypeScript or C#
- **Node Version:** 22 LTS
- **Package Manager:** npm
- **Build Tool:** NPM
- **Import Strategy:** Node.js modules

### Database & Storage
- **Primary Database:** PostgreSQL 17+ (preferred for ACID compliance and complex queries) or AWS DynamoDB (for high-scale, simple access patterns)
- **ORM:** TypeORM (for Node.js) or Entity Framework (for .NET)
- **Database Hosting:** AWS Aurora Serverless
- **Database Backups:** Daily automated
- **Asset Storage:** Amazon S3
- **CDN:** Cloudflare
- **Asset Access:** Private with signed URLs

### Testing & Quality
- **Test Framework:** Jest (Node.js) or xUnit (.NET)
- **Linting/Formatting:** ESLint + Prettier (Node.js) or RuboCop (.NET)

### Deployment & Infrastructure
- **Application Hosting:** AWS EKS Cluster
- **Hosting Region:** DEV uses us-east-1 and PROD uses sa-east-1
- **CI/CD Platform:** GitHub Actions
- **CI/CD Trigger:** Push to main/development branches
- **Tests:** Run before deployment
- **Production Environment:** main branch
- **Development Environment:** development branch
- **IaC Tool:** Terraform (latest stable version)
- **State Management:** Remote backends (S3 with DynamoDB for state locking)
- **State Encryption:** Enabled for all environments

## Infrastructure as Code (IaC)

### Terraform Configuration
- **Provider Versions:** Always lock provider versions to prevent breaking changes
- **Workspace Strategy:** Separate workspaces for dev, staging, prod environments
- **Module Organization:** Reusable modules by service/application domain
- **Code Formatting:** Always run `terraform fmt` before commits
- **Validation:** Use `terraform validate` and linting tools (`tflint`, `terrascan`)
- **Security:** Store sensitive values in AWS Secrets Manager, never hardcode

### Darwin Seguros Terraform Modules
**Available Company Modules** (prefer these over custom implementations):
- `darwin-seguros/terraform-module-sns` - SNS topic management
- `darwin-seguros/terraform-module-sns-subscription` - SNS subscription handling  
- `darwin-seguros/terraform-module-sqs` - SQS queue configuration
- `darwin-seguros/terraform-module-dynamodb` - DynamoDB table setup
- `darwin-seguros/terraform-aws-ecs` - ECS cluster deployment
- `darwin-seguros/terraform-aws-lambda` - Lambda function management

**Module Usage Strategy:**
- Always check for existing company modules before creating custom resources
- Use semantic versioning when referencing modules
- Document module configurations in project README
- Follow company module conventions for tagging and naming

### Terraform Development Workflow
- **Local Development:** Use `terraform plan` to preview changes
- **CI/CD Integration:** Automated `terraform plan` in pull requests
- **Testing:** Use `terratest` for infrastructure testing
- **Documentation:** Maintain README.md files for all modules
- **Resource Tagging:** Consistent tagging for cost management and tracking

## Logging and Observability Infrastructure

### Log Aggregation and Shipping

- **Log Shipper:** All containerized applications running in Kubernetes MUST write their logs to `stdout` and `stderr`. A log shipping agent like **Promtail** or the **Grafana Agent** is responsible for collecting these logs, enriching them with Kubernetes metadata (e.g., `pod`, `namespace`, `container`), and forwarding them to Loki.
- **Log Format:** All applications MUST output logs in a structured, newline-delimited JSON format with correlation IDs for tracing across module boundaries.
- **Log Aggregation:** Grafana Loki for log storage and querying
- **Log Visualization:** Grafana for dashboards and log exploration
- **Alerting:** Grafana alerting rules for error rate thresholds and system health

### Modular Debugging Support

**Module Boundary Logging:**
- Log all inputs and outputs at module interfaces
- Include correlation IDs for request tracing
- Structured context data for business operations
- Performance timing for module execution

**Error Context Preservation:**
- Stack trace preservation across module boundaries
- Error correlation with business context
- Failure mode classification (validation, business logic, infrastructure)
- Automated error aggregation and alerting

### .NET Core / C# Applications

- **Primary Logging Library:** **Serilog** is the standard logging library for all .NET applications. It provides robust support for structured logging and a rich ecosystem of extensions.
- **Required NuGet Packages:**
  - `Serilog.AspNetCore`: For integration with ASP.NET Core.
  - `Serilog.Sinks.Console`: To write logs to standard output.
  - `Serilog.Formatting.Compact`: Provides a clean, compact JSON formatter.
  - `Serilog.Enrichers.CorrelationId`: To automatically add a correlation ID for tracking requests.
  - `Serilog.Enrichers.Environment`: To add machine name and environment details.
  - `Serilog.Enrichers.Span`: For OpenTelemetry span correlation.

**Modular Logging Configuration:**
```csharp
builder.Host.UseSerilog((context, services, configuration) => configuration
    .ReadFrom.Configuration(context.Configuration)
    .Enrich.FromLogContext()
    .Enrich.WithProperty("service_name", "your-service-name")
    .Enrich.WithCorrelationId()
    .Enrich.WithSpan() // OpenTelemetry integration
    .Enrich.WithMachineName()
    .WriteTo.Console(new RenderedCompactJsonFormatter()));
```

### NestJS / TypeScript Applications

- **Primary Logging Library:** **`nestjs-pino`** is the standard logging library. Pino is a high-performance logger that natively produces structured JSON logs with minimal overhead.
- **Required NPM Packages:**
  - `nestjs-pino`: The core integration library for NestJS.
  - `pino-http`: Used by `nestjs-pino` for automatic request/response logging.
  - `pino-pretty`: (Optional, for development only) To make JSON logs more human-readable in local development environments.
  - `pino-opentelemetry-transport`: For OpenTelemetry trace correlation.

**Modular Logging Configuration:**
```typescript
LoggerModule.forRoot({
  pinoHttp: {
    level: process.env.NODE_ENV !== 'production' ? 'debug' : 'info',
    transport: process.env.NODE_ENV !== 'production'
      ? { target: 'pino-pretty' }
      : { target: 'pino-opentelemetry-transport' },
    customProps: (req, res) => ({
      service_name: 'your-nestjs-service',
      correlation_id: req.headers['x-correlation-id'] || generateId(),
    }),
    serializers: {
      req: (req) => ({
        method: req.method,
        path: req.url,
        client_ip: req.remoteAddress,
      }),
      res: (res) => ({
        status_code: res.statusCode,
      }),
    },
  },
});
```

## Architecture Decision Rationale

### Why TypeScript/C# Over Other Languages

**Strong Typing for Interface Contracts:**
- Compile-time verification of module interfaces
- Auto-completion and refactoring support in IDEs
- Self-documenting code with type annotations
- Reduced runtime errors through static analysis

**Ecosystem Maturity:**
- Rich dependency injection frameworks
- Comprehensive testing libraries
- Enterprise-grade logging and monitoring tools
- Large community and extensive documentation

### Why NestJS/ASP.NET Core Over Other Frameworks

**Built-in Modular Architecture:**
- Module system enforces separation of concerns
- Dependency injection container included
- Decorator-based metadata for clean interfaces
- Plugin architecture for extensibility

**Testing and Debugging Support:**
- Comprehensive testing utilities included
- Built-in health checks and metrics
- Structured logging integration
- Development tools and debugging support

### Why PostgreSQL Over Other Databases

**ACID Compliance and Reliability:**
- Strong consistency for business-critical operations
- Mature transaction management
- Proven reliability in production environments
- Excellent backup and recovery tools

**Advanced Features:**
- JSON/JSONB support for flexible schemas
- Full-text search capabilities
- Extensible with custom functions and types
- Excellent performance optimization tools

**ORM Integration:**
- First-class TypeORM and Entity Framework support
- Advanced migration and schema management
- Query optimization and debugging tools
- Connection pooling and performance monitoring

### Why Kubernetes Over Other Orchestration

**Industry Standard:**
- Widely adopted with large community
- Extensive tooling and ecosystem
- Cloud provider native support
- Standardized deployment patterns

**Modular Service Management:**
- Service discovery and load balancing
- Rolling deployments and rollbacks
- Resource management and scaling
- Health checks and self-healing

**Observability Integration:**
- Native logging and metrics collection
- Distributed tracing support
- Performance monitoring integration
- Debugging and troubleshooting tools

## Modular Architecture Support

### Dependency Injection

**NestJS Applications:**
- Use built-in IoC container with `@Injectable()` decorators
- Define interfaces for all service dependencies
- Register implementations in modules using providers
- Support for factory providers for complex initialization

**Required Packages:**
- `@nestjs/common`: Core dependency injection decorators
- `@nestjs/core`: Module and provider registration
- `reflect-metadata`: Required for decorator metadata

**.NET Core Applications:**
- Use built-in ServiceCollection for dependency registration
- Register services by interface in `Program.cs` or `Startup.cs`
- Support for singleton, scoped, and transient lifetimes
- Configuration binding for typed settings

**Required Packages:**
- `Microsoft.Extensions.DependencyInjection`: Core DI container
- `Microsoft.Extensions.Configuration`: Configuration binding
- `Microsoft.Extensions.Options`: Strongly-typed configuration

### Testing Infrastructure

**Unit Testing:**
- **NestJS**: Jest with `@nestjs/testing` for module testing
- **.NET**: xUnit with `Microsoft.Extensions.DependencyInjection` for IoC testing
- Mock frameworks: `jest.fn()` for JavaScript, `Moq` for .NET
- Test containers for integration tests with real databases

**Required Testing Packages (NestJS):**
- `jest`: Test framework
- `@nestjs/testing`: NestJS-specific testing utilities
- `supertest`: HTTP integration testing
- `testcontainers`: Docker containers for integration tests

**Required Testing Packages (.NET):**
- `xunit`: Test framework
- `xunit.runner.visualstudio`: Test runner
- `Microsoft.AspNetCore.Mvc.Testing`: Integration testing
- `Moq`: Mocking framework
- `Testcontainers`: Docker containers for integration tests

### Database and ORM Patterns

**Repository Pattern Support:**
- **TypeORM (NestJS)**: Custom repository classes with `@EntityRepository`
- **Entity Framework (.NET)**: Generic repository pattern with `DbContext`
- Interface-based repositories for easy mocking and replacement
- Unit of Work pattern for transaction management

**Migration and Schema Management:**
- **TypeORM**: Code-first migrations with TypeScript
- **Entity Framework**: Code-first migrations with .NET CLI
- Automated migration scripts in CI/CD pipeline
- Database seeding for development and testing

### Debugging and Observability

**Distributed Tracing:**
- OpenTelemetry integration for both NestJS and .NET Core
- Correlation ID propagation across service boundaries
- Jaeger or Zipkin for trace visualization
- Custom spans for business logic boundaries

**Required Observability Packages (NestJS):**
- `@opentelemetry/api`: OpenTelemetry API
- `@opentelemetry/sdk-node`: Node.js SDK
- `@opentelemetry/instrumentation-http`: HTTP instrumentation
- `@opentelemetry/instrumentation-nestjs-core`: NestJS instrumentation

**Required Observability Packages (.NET):**
- `OpenTelemetry`: Core OpenTelemetry library
- `OpenTelemetry.Extensions.Hosting`: .NET hosting integration
- `OpenTelemetry.Instrumentation.AspNetCore`: ASP.NET Core instrumentation
- `OpenTelemetry.Instrumentation.Http`: HTTP client instrumentation

**Performance Monitoring:**
- Application metrics collection (response times, error rates)
- Custom business metrics (order processing time, user engagement)
- Memory and CPU profiling in development
- Database query performance monitoring

**Health Checks:**
- Built-in health check endpoints for Kubernetes readiness/liveness
- Dependency health checks (database, external APIs)
- Circuit breaker patterns for external service failures
- Graceful shutdown handling

**Required Health Check Packages (NestJS):**
- `@nestjs/terminus`: Health check module
- `@godaddy/terminus`: Graceful shutdown

**Required Health Check Packages (.NET):**
- `Microsoft.Extensions.Diagnostics.HealthChecks`: Core health checks
- `AspNetCore.HealthChecks.SqlServer`: SQL Server health checks
- `AspNetCore.HealthChecks.Redis`: Redis health checks

## Development Workflow Support

### Code Quality Tools

**Static Analysis:**
- **ESLint + Prettier** for JavaScript/TypeScript formatting and linting
- **SonarQube** for code quality analysis and technical debt tracking
- **Husky + lint-staged** for pre-commit hooks
- **EditorConfig** for consistent formatting across IDEs

**Security Scanning:**
- **npm audit** for Node.js dependency vulnerabilities
- **NuGet Package Vulnerability Scanner** for .NET dependencies
- **OWASP Dependency Check** in CI/CD pipeline
- **CodeQL** for semantic code analysis

### Local Development

**Container Development:**
- Docker Compose for local multi-service development
- Development containers with VS Code devcontainer support
- Hot reload support for both NestJS and .NET Core
- Local database seeding scripts

**Environment Management:**
- **dotenv** files for local configuration
- Environment-specific configuration validation
- Secrets management with Azure Key Vault or AWS Secrets Manager
- Configuration schema validation at startup

### CI/CD Integration

**Build Pipeline:**
- Multi-stage Docker builds for optimized production images
- Parallel test execution for faster feedback
- Code coverage reporting with minimum thresholds
- Automated dependency updates with Dependabot

**Deployment Strategy:**
- Blue-green deployments for zero-downtime releases
- Canary releases for gradual rollouts
- Rollback capabilities with database migration compatibility
- Infrastructure as Code with Terraform (preferred) or AWS CDK

### Terraform CI/CD Integration

**Pipeline Requirements:**
- Terraform plan execution in CI pipelines before infrastructure changes
- Automated Terraform apply for approved changes
- Infrastructure testing with `terratest` in CI/CD
- Security scanning with `tfsec` and `checkov`
