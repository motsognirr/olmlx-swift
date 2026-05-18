import Foundation

public struct TestRunner {
    public static func run(tests: [String: () throws -> Void]) {
        var passed = 0
        var failed = 0
        var errors: [(String, String)] = []

        print("Running \(tests.count) tests...")
        print(String(repeating: "-", count: 50))

        for (name, testFn) in tests.sorted(by: { $0.key < $1.key }) {
            do {
                try testFn()
                passed += 1
                print("  PASS  \(name)")
            } catch {
                failed += 1
                let message = "\(error)"
                errors.append((name, message))
                print("  FAIL  \(name)")
                print("        \(message)")
            }
        }

        print(String(repeating: "-", count: 50))
        print("Results: \(passed) passed, \(failed) failed, \(passed + failed) total")

        if failed > 0 {
            print("\nFailures:")
            for (name, msg) in errors {
                print("  - \(name): \(msg)")
            }
            exit(1)
        }
    }
}

public struct TestError: Error, CustomStringConvertible {
    public let message: String
    public let file: StaticString
    public let line: UInt

    public init(_ message: String, file: StaticString = #fileID, line: UInt = #line) {
        self.message = message
        self.file = file
        self.line = line
    }

    public var description: String {
        "\(file):\(line) - \(message)"
    }
}

public func expect(
    _ condition: Bool,
    _ message: String = "expected true",
    file: StaticString = #fileID,
    line: UInt = #line
) throws {
    guard condition else { throw TestError(message, file: file, line: line) }
}

public func expectEqual<T: Equatable>(
    _ lhs: T,
    _ rhs: T,
    _ message: String? = nil,
    file: StaticString = #fileID,
    line: UInt = #line
) throws {
    guard lhs == rhs else {
        let msg = message ?? "expected \(rhs), got \(lhs)"
        throw TestError(msg, file: file, line: line)
    }
}

public func expectNil<T>(
    _ value: T?,
    _ message: String? = nil,
    file: StaticString = #fileID,
    line: UInt = #line
) throws {
    guard value == nil else {
        let msg = message ?? "expected nil, got \(value!)"
        throw TestError(msg, file: file, line: line)
    }
}

public func expectNotNil<T>(
    _ value: T?,
    _ message: String? = nil,
    file: StaticString = #fileID,
    line: UInt = #line
) throws {
    guard value != nil else {
        let msg = message ?? "expected non-nil value"
        throw TestError(msg, file: file, line: line)
    }
}
