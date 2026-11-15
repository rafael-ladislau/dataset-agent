## Coding style best practices

- **Consistent Naming Conventions**: Establish and follow naming conventions for variables, functions, classes, and files across the codebase
- **Automated Formatting**: Maintain consistent code style (indenting, line breaks, etc.)
- **Meaningful Names**: Choose descriptive names that reveal intent; avoid abbreviations and single-letter variables except in narrow contexts
- **Small, Focused Functions**: Keep functions small and focused on a single task for better readability and testability
- **Consistent Indentation**: Use consistent indentation (spaces or tabs) and configure your editor/linter to enforce it
- **Remove Dead Code**: Delete unused code, commented-out blocks, and imports rather than leaving them as clutter
- **Backward compatability only when required:** Unless specifically instructed otherwise, assume you do not need to write additional code logic to handle backward compatability.
- **DRY Principle**: Avoid duplication by extracting common logic into reusable functions or modules

## General Formatting

### Indentation
- Use 2 spaces for indentation (never tabs)
- Maintain consistent indentation throughout files
- Align nested structures for readability

### Naming Conventions
- **Methods and Variables**: Use snake_case (e.g., `user_profile`, `calculate_total`)
- **Classes and Modules**: Use PascalCase (e.g., `UserProfile`, `PaymentProcessor`)
- **Constants**: Use UPPER_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`)

### String Formatting
- Use single quotes for strings: `'Hello World'`
- Use double quotes only when interpolation is needed
- Use template literals for multi-line strings or complex interpolation

**Examples:**
```typescript
// ✅ Good: Single quotes for simple strings
const message = 'User authentication successful';
const errorType = 'VALIDATION_ERROR';

// ✅ Good: Template literals for interpolation
const logMessage = `User ${userId} attempted login at ${timestamp}`;
const apiUrl = `${baseUrl}/api/v1/users/${userId}`;

// ✅ Good: Template literals for multi-line
const emailTemplate = `
  <h1>Welcome ${userName}!</h1>
  <p>Thank you for joining our platform.</p>
  <p>Your account ID is: ${accountId}</p>
`;

// ❌ Bad: Double quotes without interpolation
const badMessage = "User authentication successful";

// ❌ Bad: String concatenation instead of template literals
const badUrl = baseUrl + '/api/v1/users/' + userId;
```

### Code Comments
- Add brief comments above non-obvious business logic
- Document complex algorithms or calculations
- Explain the "why" behind implementation choices
- Never remove existing comments unless removing the associated code
- Update comments when modifying code to maintain accuracy
- Keep comments concise and relevant

### Interface Documentation
- Document public interfaces with clear purpose and usage examples
- Specify input/output types and expected behavior
- Include error conditions and handling
- Provide integration examples for complex modules

**Example Interface Documentation:**
```typescript
/**
 * Handles user authentication and session management.
 * 
 * @example
 * const auth = new UserAuthenticator(userRepo, tokenService);
 * const result = await auth.authenticate('user@example.com', 'password');
 * if (result.success) {
 *   console.log('User authenticated:', result.user.id);
 * }
 */
interface UserAuthenticator {
  /**
   * Authenticates a user with email and password.
   * 
   * @param email - User's email address
   * @param password - Plain text password
   * @returns Promise resolving to authentication result
   * @throws AuthenticationError when credentials are invalid
   * @throws ServiceUnavailableError when auth service is down
   */
  authenticate(email: string, password: string): Promise<AuthResult>;
}
```

## Modular Code Organization

### File Structure for Modules

Organize code to support black box architecture:

```
src/
├── modules/
│   ├── user/
│   │   ├── index.ts              # Public interface exports
│   │   ├── user.interface.ts     # Type definitions
│   │   ├── user.service.ts       # Implementation
│   │   ├── user.repository.ts    # Data access
│   │   └── user.test.ts         # Interface tests
│   ├── payment/
│   │   ├── index.ts
│   │   ├── payment.interface.ts
│   │   ├── stripe.service.ts     # Stripe implementation
│   │   ├── paypal.service.ts     # PayPal implementation
│   │   └── payment.test.ts
│   └── notification/
│       ├── index.ts
│       ├── notification.interface.ts
│       ├── email.service.ts
│       ├── sms.service.ts
│       └── notification.test.ts
└── shared/
    ├── types/
    ├── utils/
    └── config/
```

### Module Index Pattern

Expose only necessary interfaces through index files:

```typescript
// ✅ Good: modules/user/index.ts
export { UserService } from './user.service';
export { UserRepository } from './user.repository';
export type { User, UserCreateData, UserUpdateData } from './user.interface';

// Don't export internal implementation details
// ❌ Don't export: UserValidator, PasswordHasher, etc.
```

### Interface-First Class Design

```typescript
// ✅ Good: Define interface first
interface PaymentProcessor {
  processPayment(request: PaymentRequest): Promise<PaymentResult>;
  refundPayment(paymentId: string): Promise<RefundResult>;
  validatePaymentMethod(method: PaymentMethod): ValidationResult;
}

class StripePaymentProcessor implements PaymentProcessor {
  constructor(
    private stripeClient: StripeClient,
    private logger: Logger,
    private config: PaymentConfig
  ) {}

  async processPayment(request: PaymentRequest): Promise<PaymentResult> {
    // Implementation details hidden
    this.logger.info('Processing payment via Stripe', { 
      amount: request.amount,
      currency: request.currency 
    });
    
    try {
      const charge = await this.stripeClient.charges.create({
        amount: request.amount,
        currency: request.currency,
        source: request.paymentMethod.token
      });
      
      return {
        success: true,
        transactionId: charge.id,
        amount: charge.amount
      };
    } catch (error) {
      this.logger.error('Stripe payment failed', { error: error.message });
      return {
        success: false,
        error: this.mapStripeError(error)
      };
    }
  }

  // Private methods for internal logic
  private mapStripeError(error: any): PaymentError {
    // Convert Stripe-specific errors to generic PaymentError
  }
}
```

### Dependency Wrapper Pattern

```typescript
// ✅ Good: Wrap external dependencies
interface EmailProvider {
  sendEmail(to: string, subject: string, body: string): Promise<void>;
  sendTemplateEmail(to: string, templateId: string, data: any): Promise<void>;
}

class SendGridEmailProvider implements EmailProvider {
  constructor(
    private sendGridClient: SendGridClient,
    private logger: Logger
  ) {}

  async sendEmail(to: string, subject: string, body: string): Promise<void> {
    try {
      await this.sendGridClient.send({
        to,
        from: this.config.fromEmail,
        subject,
        html: body
      });
      
      this.logger.info('Email sent successfully', { to, subject });
    } catch (error) {
      this.logger.error('Failed to send email', { to, subject, error });
      throw new EmailDeliveryError('Failed to send email', error);
    }
  }
}

// ❌ Bad: Direct dependency usage
class BadNotificationService {
  async sendWelcomeEmail(userEmail: string) {
    // Directly using SendGrid - hard to test and replace
    const sgMail = require('@sendgrid/mail');
    sgMail.setApiKey(process.env.SENDGRID_API_KEY);
    
    await sgMail.send({
      to: userEmail,
      from: 'noreply@example.com',
      subject: 'Welcome!',
      html: '<h1>Welcome to our platform!</h1>'
    });
  }
}
```

### Configuration Management

```typescript
// ✅ Good: Configuration interfaces
interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl: boolean;
  connectionTimeout: number;
  maxConnections: number;
}

interface PaymentConfig {
  stripeApiKey: string;
  webhookSecret: string;
  defaultCurrency: string;
  maxRetries: number;
}

class ConfigurationService {
  getDatabaseConfig(): DatabaseConfig {
    return {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'app_db',
      username: process.env.DB_USER || 'app_user',
      password: process.env.DB_PASSWORD || '',
      ssl: process.env.DB_SSL === 'true',
      connectionTimeout: parseInt(process.env.DB_TIMEOUT || '30000'),
      maxConnections: parseInt(process.env.DB_MAX_CONNECTIONS || '10')
    };
  }

  getPaymentConfig(): PaymentConfig {
    const apiKey = process.env.STRIPE_API_KEY;
    if (!apiKey) {
      throw new ConfigurationError('STRIPE_API_KEY is required');
    }
    
    return {
      stripeApiKey: apiKey,
      webhookSecret: process.env.STRIPE_WEBHOOK_SECRET || '',
      defaultCurrency: process.env.DEFAULT_CURRENCY || 'USD',
      maxRetries: parseInt(process.env.PAYMENT_MAX_RETRIES || '3')
    };
  }
}

// ❌ Bad: Scattered configuration access
class BadPaymentService {
  async processPayment() {
    // Configuration scattered throughout code
    const apiKey = process.env.STRIPE_API_KEY;
    const maxRetries = parseInt(process.env.MAX_RETRIES || '3');
    const currency = process.env.CURRENCY || 'USD';
  }
}
```
