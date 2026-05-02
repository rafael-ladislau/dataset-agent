## JavaScript/TypeScript Code Standards

JavaScript/TypeScript specific coding standards for Agent OS projects, emphasizing modular architecture and black box design principles.

## Language Standards

### TypeScript Usage
- **Always use TypeScript** for new projects and files
- Enable strict mode in `tsconfig.json`
- Use explicit type annotations for public interfaces
- Prefer type inference for internal implementation details

```typescript
// ✅ Good: Explicit types for interfaces
interface UserRepository {
  findById(id: string): Promise<User | null>;
  create(userData: CreateUserData): Promise<User>;
  update(id: string, updates: Partial<User>): Promise<User>;
}

// ✅ Good: Type inference for implementation
class PostgresUserRepository implements UserRepository {
  async findById(id: string) { // Return type inferred as Promise<User | null>
    const result = await this.db.query('SELECT * FROM users WHERE id = $1', [id]);
    return result.rows[0] ? this.mapToUser(result.rows[0]) : null;
  }
}
```

### Variable Declarations
- Use `const` by default
- Use `let` when reassignment is necessary
- Never use `var`

```typescript
// ✅ Good: const for immutable values
const API_BASE_URL = 'https://api.example.com';
const userService = new UserService(userRepository, logger);

// ✅ Good: let for reassignment
let retryCount = 0;
while (retryCount < MAX_RETRIES) {
  // ... retry logic
  retryCount++;
}

// ❌ Bad: var usage
var userId = request.params.id; // Never use var
```

### Function Declarations
- Use arrow functions for short, inline functions
- Use function declarations for main module functions
- Use method syntax in classes

```typescript
// ✅ Good: Arrow functions for callbacks and short operations
const users = await Promise.all(
  userIds.map(async (id) => await userRepository.findById(id))
);

const isValidEmail = (email: string): boolean => 
  /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

// ✅ Good: Function declarations for main functions
async function processPayment(paymentData: PaymentData): Promise<PaymentResult> {
  // Implementation
}

// ✅ Good: Method syntax in classes
class UserService {
  async createUser(userData: CreateUserData): Promise<User> {
    // Implementation
  }
}
```

## Modular Architecture Patterns

### Module Structure
- Each module should have a clear, single responsibility
- Export only necessary interfaces through index files
- Keep implementation details private

```typescript
// ✅ Good: Clear module structure
// src/modules/user/index.ts
export { UserService } from './user.service';
export type { User, CreateUserData, UpdateUserData } from './user.types';
export type { UserRepository } from './user.repository.interface';

// Don't export implementation classes
// ❌ Don't export: PostgresUserRepository, UserValidator, etc.
```

### Interface-First Design

```typescript
// ✅ Good: Define interfaces before implementation
interface NotificationService {
  sendEmail(to: string, subject: string, body: string): Promise<void>;
  sendSMS(to: string, message: string): Promise<void>;
  sendPushNotification(userId: string, title: string, body: string): Promise<void>;
}

interface EmailProvider {
  send(email: EmailMessage): Promise<void>;
  sendTemplate(templateId: string, to: string, data: any): Promise<void>;
}

class EmailNotificationService implements NotificationService {
  constructor(
    private emailProvider: EmailProvider,
    private smsProvider: SMSProvider,
    private pushProvider: PushProvider
  ) {}

  async sendEmail(to: string, subject: string, body: string): Promise<void> {
    await this.emailProvider.send({ to, subject, body });
  }
}
```

### Error Handling Patterns

```typescript
// ✅ Good: Custom error classes with proper inheritance
abstract class DomainError extends Error {
  abstract readonly code: string;
  abstract readonly statusCode: number;

  constructor(message: string, public readonly cause?: Error) {
    super(message);
    this.name = this.constructor.name;
  }
}

class ValidationError extends DomainError {
  readonly code = 'VALIDATION_ERROR';
  readonly statusCode = 400;
}

class UserNotFoundError extends DomainError {
  readonly code = 'USER_NOT_FOUND';
  readonly statusCode = 404;
}

// ✅ Good: Structured error handling
class UserService {
  async getUserById(id: string): Promise<User> {
    try {
      this.validateUserId(id);
      
      const user = await this.userRepository.findById(id);
      if (!user) {
        throw new UserNotFoundError(`User with ID ${id} not found`);
      }
      
      return user;
    } catch (error) {
      this.logger.error('Failed to get user', { userId: id, error: error.message });
      
      // Re-throw domain errors, wrap infrastructure errors
      if (error instanceof DomainError) {
        throw error;
      }
      
      throw new DatabaseConnectionError('Failed to retrieve user', error);
    }
  }

  private validateUserId(id: string): void {
    if (!id || typeof id !== 'string' || id.trim().length === 0) {
      throw new ValidationError('User ID is required and must be a non-empty string');
    }
  }
}
```

## Testing Patterns

### Unit Testing with Jest

```typescript
// ✅ Good: Test interfaces, not implementations
describe('UserService', () => {
  let userService: UserService;
  let mockUserRepository: jest.Mocked<UserRepository>;
  let mockLogger: jest.Mocked<Logger>;

  beforeEach(() => {
    mockUserRepository = {
      findById: jest.fn(),
      findByEmail: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn()
    };

    mockLogger = {
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn()
    };

    userService = new UserService(mockUserRepository, mockLogger);
  });

  describe('createUser', () => {
    it('should create user successfully with valid data', async () => {
      // Arrange
      const userData: CreateUserData = {
        email: 'test@example.com',
        password: 'securePassword123',
        name: 'Test User'
      };
      
      const expectedUser: User = {
        id: 'user-123',
        email: userData.email,
        name: userData.name,
        createdAt: new Date()
      };
      
      mockUserRepository.findByEmail.mockResolvedValue(null);
      mockUserRepository.create.mockResolvedValue(expectedUser);

      // Act
      const result = await userService.createUser(userData);

      // Assert
      expect(result).toEqual(expectedUser);
      expect(mockUserRepository.findByEmail).toHaveBeenCalledWith(userData.email);
      expect(mockUserRepository.create).toHaveBeenCalledWith(userData);
    });

    it('should throw UserAlreadyExistsError when email is taken', async () => {
      // Arrange
      const userData: CreateUserData = {
        email: 'existing@example.com',
        password: 'newPassword123',
        name: 'Test User'
      };
      
      const existingUser: User = {
        id: 'existing-user',
        email: userData.email,
        name: 'Existing User',
        createdAt: new Date()
      };
      
      mockUserRepository.findByEmail.mockResolvedValue(existingUser);

      // Act & Assert
      await expect(userService.createUser(userData))
        .rejects
        .toThrow(UserAlreadyExistsError);
        
      expect(mockUserRepository.create).not.toHaveBeenCalled();
    });
  });
});
```

### Testing Patterns for Modular Code

```typescript
// ✅ Good: Test the interface, not implementation
describe('UserService', () => {
  let userService: UserService;
  let mockUserRepository: jest.Mocked<UserRepository>;
  let mockLogger: jest.Mocked<Logger>;

  beforeEach(() => {
    mockUserRepository = {
      create: jest.fn(),
      findByEmail: jest.fn(),
      findById: jest.fn(),
      update: jest.fn(),
      delete: jest.fn()
    };
    
    mockLogger = {
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn()
    };
    
    userService = new UserService(mockUserRepository, mockLogger);
  });

  describe('createUser', () => {
    it('should create user successfully with valid data', async () => {
      // Arrange
      const userData: CreateUserData = {
        email: 'test@example.com',
        password: 'securePassword123',
        name: 'Test User'
      };
      
      const expectedUser: User = {
        id: 'user-123',
        email: userData.email,
        name: userData.name,
        createdAt: new Date()
      };
      
      mockUserRepository.findByEmail.mockResolvedValue(null);
      mockUserRepository.create.mockResolvedValue(expectedUser);

      // Act
      const result = await userService.createUser(userData);

      // Assert
      expect(result).toEqual(expectedUser);
      expect(mockUserRepository.findByEmail).toHaveBeenCalledWith(userData.email);
      expect(mockUserRepository.create).toHaveBeenCalledWith(userData);
      expect(mockLogger.info).toHaveBeenCalledWith(
        'User created successfully',
        { userId: expectedUser.id, email: expectedUser.email }
      );
    });

    it('should throw UserAlreadyExistsError when email is taken', async () => {
      // Arrange
      const userData: CreateUserData = {
        email: 'existing@example.com',
        password: 'securePassword123',
        name: 'Test User'
      };
      
      const existingUser: User = {
        id: 'existing-user',
        email: userData.email,
        name: 'Existing User',
        createdAt: new Date()
      };
      
      mockUserRepository.findByEmail.mockResolvedValue(existingUser);

      // Act & Assert
      await expect(userService.createUser(userData))
        .rejects
        .toThrow(UserAlreadyExistsError);
        
      expect(mockUserRepository.create).not.toHaveBeenCalled();
    });
  });
});
```

## Anti-Patterns to Avoid

### Common JavaScript Anti-Patterns

```typescript
// ❌ Bad: Callback hell
function processUser(userId: string, callback: (error: Error | null, result?: any) => void) {
  getUserById(userId, (error, user) => {
    if (error) return callback(error);
    
    getUserPreferences(userId, (error, preferences) => {
      if (error) return callback(error);
      
      processUserData(user, preferences, (error, result) => {
        if (error) return callback(error);
        callback(null, result);
      });
    });
  });
}

// ✅ Good: Async/await
async function processUser(userId: string): Promise<ProcessedUserData> {
  const user = await this.userRepository.findById(userId);
  const preferences = await this.preferencesService.getPreferences(userId);
  return await this.processUserData(user, preferences);
}

// ❌ Bad: Mutating function parameters
function updateUser(user: User, updates: Partial<User>): User {
  // Mutating the original object
  Object.assign(user, updates);
  user.updatedAt = new Date();
  return user;
}

// ✅ Good: Immutable updates
function updateUser(user: User, updates: Partial<User>): User {
  return {
    ...user,
    ...updates,
    updatedAt: new Date()
  };
}

// ❌ Bad: Inconsistent error handling
async function fetchUserData(id: string) {
  try {
    const user = await userRepository.findById(id);
    return user;
  } catch (error) {
    console.log('Error:', error); // Inconsistent logging
    return null; // Swallowing errors
  }
}

// ✅ Good: Consistent error handling
async function fetchUserData(id: string): Promise<User> {
  try {
    const user = await this.userRepository.findById(id);
    if (!user) {
      throw new UserNotFoundError(`User with ID ${id} not found`);
    }
    return user;
  } catch (error) {
    this.logger.error('Failed to fetch user data', { userId: id, error: error.message });
    throw error; // Re-throw for proper error propagation
  }
}
```

