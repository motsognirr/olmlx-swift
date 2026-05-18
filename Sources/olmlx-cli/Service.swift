import ArgumentParser
import Foundation

private let serviceLabel = "com.olmlx"

private func launchAgentPlistURL() -> URL {
    URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Library/LaunchAgents/\(serviceLabel).plist")
}

private func guiDomain() -> String { "gui/\(getuid())" }

private func guiTarget(_ label: String) -> String { "\(guiDomain())/\(label)" }

private struct LaunchctlResult {
    let exitCode: Int32
    let stdout: String
    let stderr: String
}

@discardableResult
private func runLaunchctl(_ args: [String]) throws -> LaunchctlResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
    process.arguments = args
    let outPipe = Pipe()
    let errPipe = Pipe()
    process.standardOutput = outPipe
    process.standardError = errPipe
    try process.run()
    process.waitUntilExit()
    let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
    return LaunchctlResult(
        exitCode: process.terminationStatus,
        stdout: String(data: outData, encoding: .utf8) ?? "",
        stderr: String(data: errData, encoding: .utf8) ?? ""
    )
}

private func resolvedExecutablePath() -> String {
    let raw: String
    if let path = Bundle.main.executablePath, !path.isEmpty {
        raw = path
    } else {
        raw = CommandLine.arguments.first ?? "olmlx"
    }
    let url: URL =
        raw.hasPrefix("/")
        ? URL(fileURLWithPath: raw)
        : URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(raw)
    return url.resolvingSymlinksInPath().path
}

private func renderPlistData(
    executable: String,
    environment: [String: String],
    standardOutPath: String,
    standardErrorPath: String,
    workingDirectory: String,
    keepAlive: Bool
) throws -> Data {
    var plist: [String: Any] = [
        "Label": serviceLabel,
        "ProgramArguments": [executable, "serve"],
        "RunAtLoad": true,
        "KeepAlive": keepAlive,
        "StandardOutPath": standardOutPath,
        "StandardErrorPath": standardErrorPath,
        "WorkingDirectory": workingDirectory,
    ]
    if !environment.isEmpty {
        plist["EnvironmentVariables"] = environment
    }
    return try PropertyListSerialization.data(
        fromPropertyList: plist,
        format: .xml,
        options: 0
    )
}

private func parseEnvOption(_ raw: String) throws -> (String, String) {
    guard let eq = raw.firstIndex(of: "=") else {
        throw ValidationError("--env value must be in KEY=VALUE form, got '\(raw)'")
    }
    let key = String(raw[..<eq]).trimmingCharacters(in: .whitespaces)
    guard !key.isEmpty else {
        throw ValidationError("--env key cannot be empty")
    }
    let value = String(raw[raw.index(after: eq)...])
    return (key, value)
}

// `launchctl print` output is indented `key = value` lines. Return the value for
// the first matching key, trimmed.
private func extractField(_ output: String, key: String) -> String? {
    for line in output.split(separator: "\n") {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let eq = trimmed.firstIndex(of: "=") else { continue }
        let lhs = trimmed[..<eq].trimmingCharacters(in: .whitespaces)
        if lhs == key {
            return trimmed[trimmed.index(after: eq)...].trimmingCharacters(in: .whitespaces)
        }
    }
    return nil
}

struct Service: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Launchd service management",
        subcommands: [ServiceInstall.self, ServiceStatus.self, ServiceUninstall.self]
    )
}

struct ServiceInstall: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "install",
        abstract: "Install olmlx as a launchd LaunchAgent and start it"
    )

    @Option(name: .long, help: "Path to the olmlx binary (default: current executable)")
    var executable: String?

    @Option(name: .long, help: "OLMLX_HOST for the service")
    var host: String?

    @Option(name: .long, help: "OLMLX_PORT for the service")
    var port: Int?

    @Option(
        name: [.customShort("e"), .customLong("env")],
        help: "Additional environment variable in KEY=VALUE form (repeatable)"
    )
    var env: [String] = []

    @Option(name: .long, help: "Directory for service stdout/stderr logs")
    var logDir: String?

    @Flag(name: .long, inversion: .prefixedNo, help: "Restart the service if it exits")
    var keepAlive: Bool = true

    @Flag(name: .long, help: "Replace an existing installation instead of failing")
    var force: Bool = false

    @Flag(name: .customLong("no-start"), help: "Write the plist but do not bootstrap it")
    var noStart: Bool = false

    mutating func run() throws {
        let exe =
            executable.map { ($0 as NSString).expandingTildeInPath }
            ?? resolvedExecutablePath()
        guard FileManager.default.isExecutableFile(atPath: exe) else {
            throw ValidationError("Executable not found or not runnable: \(exe)")
        }

        var environment: [String: String] = [
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        ]
        if let host { environment["OLMLX_HOST"] = host }
        if let port {
            guard (1...65535).contains(port) else {
                throw ValidationError("--port must be between 1 and 65535")
            }
            environment["OLMLX_PORT"] = String(port)
        }
        for raw in env {
            let (key, value) = try parseEnvOption(raw)
            environment[key] = value
        }

        let logDirURL: URL =
            logDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".olmlx/logs")
        try FileManager.default.createDirectory(at: logDirURL, withIntermediateDirectories: true)

        let plistURL = launchAgentPlistURL()
        try FileManager.default.createDirectory(
            at: plistURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        if FileManager.default.fileExists(atPath: plistURL.path) {
            guard force else {
                throw ValidationError(
                    "\(plistURL.path) already exists. Re-run with --force, "
                        + "or run `olmlx service uninstall` first.")
            }
            // Boot out any running instance so the rewrite doesn't leave a stale process attached
            // to the old plist contents.
            _ = try? runLaunchctl(["bootout", guiTarget(serviceLabel)])
        }

        let data = try renderPlistData(
            executable: exe,
            environment: environment,
            standardOutPath: logDirURL.appendingPathComponent("olmlx.out.log").path,
            standardErrorPath: logDirURL.appendingPathComponent("olmlx.err.log").path,
            workingDirectory: NSHomeDirectory(),
            keepAlive: keepAlive
        )
        try data.write(to: plistURL, options: .atomic)
        print("Wrote \(plistURL.path)")

        guard !noStart else {
            print("Skipping bootstrap (--no-start). Load with:")
            print("  launchctl bootstrap \(guiDomain()) \(plistURL.path)")
            return
        }

        let bootstrap = try runLaunchctl(["bootstrap", guiDomain(), plistURL.path])
        guard bootstrap.exitCode == 0 else {
            let detail = [bootstrap.stdout, bootstrap.stderr]
                .filter { !$0.isEmpty }.joined(separator: "\n")
            throw ValidationError(
                "launchctl bootstrap failed (exit \(bootstrap.exitCode)):\n\(detail)")
        }
        print("Bootstrapped \(guiTarget(serviceLabel))")
        print("Tail errors with:")
        print("  tail -f \(logDirURL.appendingPathComponent("olmlx.err.log").path)")
    }
}

struct ServiceStatus: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Check launchd service status"
    )

    mutating func run() throws {
        let plistURL = launchAgentPlistURL()
        let plistExists = FileManager.default.fileExists(atPath: plistURL.path)

        let result = try runLaunchctl(["print", guiTarget(serviceLabel)])
        guard result.exitCode == 0 else {
            if plistExists {
                print("Service installed but not loaded.")
                print("Plist: \(plistURL.path)")
                print("Load with: olmlx service install --force")
            } else {
                print("Service not installed. Run `olmlx service install`.")
            }
            return
        }

        print("Service: \(serviceLabel)")
        print("Plist: \(plistURL.path)")
        if let state = extractField(result.stdout, key: "state") {
            print("State: \(state)")
        }
        if let pid = extractField(result.stdout, key: "pid") {
            print("PID: \(pid)")
        } else {
            print("PID: (not running)")
        }
        if let lastExit = extractField(result.stdout, key: "last exit code") {
            print("Last exit code: \(lastExit)")
        }
    }
}

struct ServiceUninstall: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "uninstall",
        abstract: "Stop the service and remove its launchd plist"
    )

    @Flag(name: .long, help: "Don't fail if the service isn't installed")
    var ifExists: Bool = false

    mutating func run() throws {
        let plistURL = launchAgentPlistURL()
        let plistExists = FileManager.default.fileExists(atPath: plistURL.path)
        if !plistExists && !ifExists {
            throw ValidationError("No service installed at \(plistURL.path)")
        }

        let bootout = try runLaunchctl(["bootout", guiTarget(serviceLabel)])
        if bootout.exitCode != 0 {
            // launchctl bootout is noisy when the target isn't loaded; treat
            // "not loaded" as benign so uninstall is idempotent.
            let stderr = bootout.stderr.lowercased()
            let benign =
                stderr.contains("could not find")
                || stderr.contains("no such process")
                || stderr.isEmpty
            if !benign {
                let detail = [bootout.stdout, bootout.stderr]
                    .filter { !$0.isEmpty }.joined(separator: "\n")
                print("Warning: launchctl bootout exit \(bootout.exitCode):\n\(detail)")
            }
        } else {
            print("Booted out \(guiTarget(serviceLabel))")
        }

        if plistExists {
            try FileManager.default.removeItem(at: plistURL)
            print("Removed \(plistURL.path)")
        }
    }
}
