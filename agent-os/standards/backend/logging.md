## Application Logging Best Practices

These best practices ensure our logs are effective, secure, and performant. Following them will significantly improve our ability to diagnose issues in production.

### Guiding Principles

- **Log in JSON format**: Always use structured JSON logging. Plain text logs are considered a legacy practice and should be avoided.
- **Write to `stdout`/`stderr`**: Let the container orchestration environment and log shippers handle log file management and routing.
- **Provide Context**: The saddest log line is one without context. Always include relevant information like a `trace_id` (Correlation ID) and key business identifiers (e.g., `order_id`, `user_id`).
- **Do Not Log Sensitive Data**: Never log passwords, API keys, tokens, or Personally Identifiable Information (PII). Use redaction features in the logging libraries where necessary.
- **Use Correct Log Levels**:
  - `debug`: Detailed information for diagnosing issues during development. Should be disabled in production.
  - `info`: Standard messages tracking the normal flow of the application (e.g., service started, request processed).
  - `warn`: Indicates a potential problem or unexpected event that does not prevent the current operation from completing (e.g., retrying an API call).
  - `error`: An error that prevents the current operation from completing but does not shut down the application (e.g., database connection failure, uncaught exception in a request handler).
  - `fatal`: A critical error that forces the application to terminate.

### JSON Log Schema

To ensure consistency and effective querying in Grafana, all JSON logs MUST adhere to the following standardized schema. Field names should be `snake_case` for cross-language consistency.

#### Standard Log Schema

| Field Name          | Type     | Description                                                                                                                              | Example                                        |
| ------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **`timestamp`**     | `string` | **Required.** ISO 8601 formatted UTC timestamp of when the event occurred.                                                               | `"2025-08-22T11:15:01.1234567Z"`               |
| **`level`**         | `string` | **Required.** Log severity level. Must be one of: `debug`, `info`, `warn`, `error`, `fatal`.                                               | `"error"`                                      |
| **`message`**       | `string` | **Required.** A human-readable description of the event.                                                                                 | `"Failed to process payment for order 12345."` |
| **`service_name`**  | `string` | **Required.** The name of the application or microservice generating the log.                                                            | `"payment-service"`                            |
| **`trace_id`**      | `string` | **Highly Recommended.** A unique identifier for tracing a single request across multiple services. Synonymous with Correlation ID.      | `"a1b2c3d4-e5f6-7890-1234-567890abcdef"`       |
| **`span_id`**       | `string` | *Optional.* A unique identifier for a specific operation within a trace.                                                                 | `"fedcba09-8765-4321-0987-654321fedcba"`       |
| `request`           | `object` | *Contextual.* Object containing details of an HTTP request.                                                                              | `{ "method": "POST", "path": "/api/payments" }`|
| `response`          | `object` | *Contextual.* Object containing details of an HTTP response.                                                                             | `{ "status_code": 500, "duration_ms": 152 }`   |
| `error`             | `object` | *Contextual.* Structured details of an exception or error.                                                                               | `{ "message": "...", "stack_trace": "..." }`   |
| `properties`        | `object` | *Optional.* A container for any other custom, application-specific contextual data.                                                      | `{ "order_id": 12345, "user_id": "usr-abc" }`   |

#### Log Examples

**Example: Informational Log**
```json
{
  "timestamp": "2025-08-22T11:15:01.123Z",
  "level": "info",
  "message": "User usr-abc successfully placed order 12345.",
  "service_name": "order-service",
  "trace_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "properties": {
    "order_id": 12345,
    "user_id": "usr-abc"
  }
}
```

**Example: HTTP Request Log (via middleware)**
```json
{
  "timestamp": "2025-08-22T11:18:30.456Z",
  "level": "info",
  "message": "HTTP POST /api/orders responded 201 in 75 ms",
  "service_name": "order-service",
  "trace_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
  "request": {
    "method": "POST",
    "path": "/api/orders",
    "client_ip": "192.168.1.100"
  },
  "response": {
    "status_code": 201,
    "duration_ms": 75
  }
}
```

**Example: Error Log with Exception**
```json
{
  "timestamp": "2025-08-22T11:20:05.789Z",
  "level": "error",
  "message": "Failed to connect to the database.",
  "service_name": "inventory-service",
  "trace_id": "c3d4e5f6-a7b8-9012-3456-7890abcdef12",
  "error": {
    "message": "A network-related or instance-specific error occurred while establishing a connection to SQL Server. The server was not found or was not accessible.",
    "type": "System.Data.SqlClient.SqlException",
    "stack_trace": "at System.Data.SqlClient.SqlInternalConnectionTds.TryOpen... (full stack trace here)"
  }
}
```

### .NET Core with Serilog Implementation

Configure Serilog in `Program.cs` to write JSON to the console and automatically enrich logs.

**Example `Program.cs` Configuration:**
```csharp
using Serilog;
using Serilog.Formatting.Compact;

// ... other using statements

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((context, services, configuration) => configuration
    .ReadFrom.Configuration(context.Configuration)
    .Enrich.FromLogContext()
    .Enrich.WithProperty("service_name", "your-service-name") // Set your service name
    .Enrich.WithCorrelationId() // Adds trace_id
    .Enrich.WithMachineName()
    .WriteTo.Console(new RenderedCompactJsonFormatter())); // Use JSON formatter

// Add HttpContextAccessor for CorrelationId enricher
builder.Services.AddHttpContextAccessor();

var app = builder.Build();

// This middleware logs all incoming HTTP requests automatically
app.UseSerilogRequestLogging();

// ... rest of your application setup
app.Run();
```

**Example Application-Specific Logging:**
```csharp
public class OrderService
{
    private readonly ILogger<OrderService> _logger;

    public OrderService(ILogger<OrderService> logger)
    {
        _logger = logger;
    }

    public async Task<Order> CreateOrderAsync(CreateOrderRequest request)
    {
        _logger.LogInformation("Creating order for user {UserId} with {ItemCount} items", 
            request.UserId, request.Items.Count);

        try
        {
            var order = await ProcessOrderAsync(request);
            
            _logger.LogInformation("Successfully created order {OrderId} for user {UserId}", 
                order.Id, request.UserId);
            
            return order;
        }
        catch (PaymentProcessingException ex)
        {
            _logger.LogError(ex, "Payment processing failed for user {UserId}. Order creation aborted.", 
                request.UserId);
            throw;
        }
    }
}
```

### NestJS with nestjs-pino Implementation

Configure nestjs-pino in your AppModule to format logs according to the standard.

**Example `app.module.ts` Configuration:**
```typescript
import { Module } from '@nestjs/common';
import { LoggerModule } from 'nestjs-pino';
import { randomUUID } from 'crypto';

@Module({
  imports: [
    LoggerModule.forRoot({
      pinoHttp: {
        // In production, you would want a simpler format. This is for demonstration.
        // For production, just use `level: 'info'` without prettyPrint.
        level: process.env.NODE_ENV !== 'production' ? 'debug' : 'info',
        transport: process.env.NODE_ENV !== 'production'
          ? { target: 'pino-pretty' } // Use pino-pretty for development only
          : undefined,
        
        // Define a custom serializer to format logs according to our standard
        serializers: {
          req(req) {
            return {
              method: req.method,
              path: req.url,
              client_ip: req.remoteAddress,
            };
          },
          res(res) {
            return {
              status_code: res.statusCode,
            };
          },
        },
        
        // Rename Pino's default fields to match our standard
        customProps: (req, res) => ({
          service_name: 'your-nestjs-service', // Set your service name
        }),
        messageKey: 'message',
        base: {
          // Remove default pino base properties like pid, hostname
        },
        // Generate or forward a trace_id (Correlation ID)
        genReqId: function (req, res) {
          const existingId = req.id ?? req.headers["x-correlation-id"];
          if (existingId) return existingId;
          const id = randomUUID();
          res.header('x-correlation-id', id);
          return id;
        },
        // Use custom keys for specific fields
        customLogLevel: (req, res, err) => {
            if (res.statusCode >= 500 || err) return 'error';
            if (res.statusCode >= 400) return 'warn';
            return 'info';
        },
        // Map pino's level to our standard 'level' field
        customAttributeKeys: {
          reqId: 'trace_id',
          responseTime: 'duration_ms',
          level: 'level', // This ensures the level is correctly named
          req: 'request',
          res: 'response',
          err: 'error'
        },
      },
    }),
  ],
  // ... your controllers and providers
})
export class AppModule {}
```

**Example Service-Level Logging:**
```typescript
import { Injectable, Logger } from '@nestjs/common';

@Injectable()
export class OrderService {
  private readonly logger = new Logger(OrderService.name);

  async createOrder(createOrderDto: CreateOrderDto): Promise<Order> {
    this.logger.log('Creating order for user', {
      user_id: createOrderDto.userId,
      item_count: createOrderDto.items.length
    });

    try {
      const order = await this.processOrder(createOrderDto);
      
      this.logger.log('Successfully created order', {
        order_id: order.id,
        user_id: createOrderDto.userId
      });
      
      return order;
    } catch (error) {
      this.logger.error('Order creation failed', {
        user_id: createOrderDto.userId,
        error: error.message,
        stack_trace: error.stack
      });
      throw error;
    }
  }
}
```

### Querying in Grafana (LogQL)

Adhering to these standards makes querying logs simple and powerful:

**Find all errors in a specific service:**
```logql
{job="your-app-deployment"} | json | service_name="payment-service" and level="error"
```

**Trace a single request across all services:**
```logql
{job=~".+"} | json | trace_id="a1b2c3d4-e5f6-7890-1234-567890abcdef"
```

**Calculate the 95th percentile request duration for the gateway:**
```logql
quantile_over_time(0.95, {job="loki-gateway"} | json | unwrap duration_ms [1m])
```

**Find all payment-related errors in the last hour:**
```logql
{job=~".+"} | json | level="error" and message =~ "(?i)payment" | line_format "{{.timestamp}} [{{.service_name}}] {{.message}}"
```

## Logging and Observability Infrastructure

### Log Aggregation and Shipping

- **Log Shipper**: All containerized applications running in Kubernetes MUST write their logs to `stdout` and `stderr`. A log shipping agent like **Promtail** or the **Grafana Agent** is responsible for collecting these logs, enriching them with Kubernetes metadata (e.g., `pod`, `namespace`, `container`), and forwarding them to Loki.
- **Log Format**: All applications MUST output logs in a structured, newline-delimited JSON format with correlation IDs for tracing across module boundaries.
- **Log Aggregation**: Grafana Loki for log storage and querying
- **Log Visualization**: Grafana for dashboards and log exploration
- **Alerting**: Grafana alerting rules for error rate thresholds and system health

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

