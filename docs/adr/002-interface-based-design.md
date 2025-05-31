# ADR-002: Interface-Based Design

## Status

Accepted

## Context

During the refactoring of the paper revision tool from a monolithic class to a layered architecture, we identified several inconsistencies in abstraction levels across different modules:

- Some modules had clean interfaces while others directly coupled implementations
- There was inconsistent use of design patterns
- Components were instantiated directly, creating tight coupling
- The system lacked a clear extension mechanism
- Testing was challenging due to concrete dependencies

We needed a consistent approach to component design and interaction that would support the layered architecture and improve testability.

## Decision

We decided to adopt an interface-based design approach with the following principles:

1. **Interface Segregation**: Define focused interfaces for each component type
2. **Dependency Inversion**: Depend on abstractions, not concrete implementations
3. **Factory Pattern**: Use factories to create and configure components
4. **Base Classes**: Provide abstract base classes for common functionality
5. **Standardized Method Signatures**: Ensure consistent method signatures across implementations

Each major component type has a corresponding interface in its layer's `interfaces.py` file:

- Service interfaces in `src/services/interfaces.py`
- Repository interfaces in `src/repositories/interfaces.py`
- Adapter interfaces in `src/adapters/interfaces.py`

Components access other components only through their interfaces, never through concrete implementations.

## Consequences

### Positive

- **Improved testability**: Components can be tested with mock implementations
- **Clearer component boundaries**: Interfaces define clear contracts
- **Enhanced extensibility**: New implementations can be added without changing clients
- **Consistent abstraction levels**: All components follow the same design approach
- **Decoupled components**: Components depend only on abstractions

### Negative

- **Increased code size**: Interfaces add additional code
- **Implementation overhead**: Defining and maintaining interfaces takes effort
- **Potential over-abstraction**: Not all components may need interfaces
- **Learning curve**: Developers need to understand the interface-based approach

## Alternatives Considered

### Direct Dependency Injection Without Interfaces

**Pros**:
- Simpler implementation
- Less code
- No interface maintenance overhead

**Cons**:
- Components are still coupled to concrete implementations
- Testing requires more complex mocking
- Less clear component boundaries

### Service Locator Pattern

**Pros**:
- Components can request dependencies when needed
- Supports dynamic configuration

**Cons**:
- Hides dependencies, making them implicit
- Makes testing more difficult
- Can lead to runtime errors if dependencies are not registered

## Implementation

The interface-based design is implemented through the following patterns:

1. **Interface Definition**:
   ```python
   class DocumentServiceInterface:
       """Interface for document services."""
       
       def create_changes_document(self, changes, output_path):
           """Create a document detailing all changes."""
           raise NotImplementedError
   ```

2. **Implementation**:
   ```python
   class DocumentService(DocumentServiceInterface):
       """Implementation of the document service."""
       
       def create_changes_document(self, changes, output_path):
           """Create a document detailing all changes."""
           # Implementation...
   ```

3. **Factory**:
   ```python
   class ServiceFactory:
       """Factory for creating service instances."""
       
       def __init__(self, context):
           self.context = context
           self._instances = {}
           
       def get_document_service(self):
           """Get a document service instance."""
           if "document_service" not in self._instances:
               self._instances["document_service"] = DocumentService(self.context)
           return self._instances["document_service"]
   ```

4. **Client Usage**:
   ```python
   document_service = service_factory.get_document_service()
   document_path = document_service.create_changes_document(changes, output_path)
   ```

## Related Decisions

- [ADR-001: Layered Architecture](001-layered-architecture.md)
- [ADR-003: Configuration Management](003-configuration-management.md)

## References

- "Design Patterns" by Gamma, Helm, Johnson, and Vlissides
- "Clean Architecture" by Robert C. Martin
- "Dependency Injection in .NET" by Mark Seemann