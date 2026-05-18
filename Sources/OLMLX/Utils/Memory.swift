import Foundation
import Metal

public func getMetalMemory() -> (active: UInt64, cache: UInt64)? {
    guard let device = MTLCreateSystemDefaultDevice() else { return nil }
    return (UInt64(device.currentAllocatedSize), 0)
}

public func getSystemMemoryBytes() -> UInt64 {
    let processInfo = ProcessInfo.processInfo
    return processInfo.physicalMemory
}

public func isMemoryPressureHigh(limitFraction: Double = 0.75) -> Bool {
    guard let (active, _) = getMetalMemory() else { return false }
    let total = getSystemMemoryBytes()
    let limit = UInt64(Double(total) * limitFraction)
    return active > limit
}
