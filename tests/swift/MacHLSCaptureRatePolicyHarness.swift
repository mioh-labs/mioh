import Foundation

@main
struct MacHLSCaptureRatePolicyHarness {
  static func require(
    _ condition: @autoclosure () -> Bool,
    _ message: String
  ) {
    guard condition() else {
      FileHandle.standardError.write(Data((message + "\n").utf8))
      exit(1)
    }
  }

  static func main() {
    var policy = MacHLSCaptureRatePolicy()
    require(
      policy.desiredRate(
        bufferedSeconds: 6,
        targetSeconds: 60,
        isLive: false
      ) == 1,
      "capture must warm up at 1x until the first complete segment"
    )
    require(
      policy.finishWarmup(),
      "the first complete segment must finish warmup"
    )
    require(
      policy.desiredRate(
        bufferedSeconds: 6,
        targetSeconds: 60,
        isLive: false
      ) == 2,
      "VOD capture must fill at 2x after warmup"
    )
    require(
      policy.desiredRate(
        bufferedSeconds: 60,
        targetSeconds: 60,
        isLive: false
      ) == 1,
      "capture must return to 1x at the configured target"
    )
    require(
      policy.desiredRate(
        bufferedSeconds: 50,
        targetSeconds: 60,
        isLive: false
      ) == 1,
      "hysteresis must not immediately restart acceleration"
    )
    require(
      policy.desiredRate(
        bufferedSeconds: 42,
        targetSeconds: 60,
        isLive: false
      ) == 2,
      "capture may refill after the restored lead materially falls"
    )

    require(
      policy.desiredRate(
        bufferedSeconds: 0,
        targetSeconds: 60,
        isLive: true
      ) == 1,
      "live HLS cannot run ahead of the live edge"
    )

    print("Mac HLS capture rate policy passed")
  }
}
