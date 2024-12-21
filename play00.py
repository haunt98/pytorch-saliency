encoder_scales = 5
encoder_base = 64
upsampler_scales = 3
upsampler_base = 64

down = encoder_scales
modulator_sizes = []
for up in reversed(range(upsampler_scales)):
    print(f"down: {down}, up: {up}")

    upsampler_chans = upsampler_base * 2 ** (up + 1)
    encoder_chans = encoder_base * 2**down
    print(f"upsampler_chans: {upsampler_chans}, encoder_chans: {encoder_chans}")

    inc = upsampler_chans if down != encoder_scales else encoder_chans
    modulator_sizes.append(inc)

    in_channels_0 = inc
    out_channels_0 = upsampler_chans // 2
    passthrough_channels_0 = encoder_chans // 2
    print(
        f"UNetUpsampler: in_channels: {in_channels_0}, out_channels: {out_channels_0}, passthrough_channels: {passthrough_channels_0}"
    )
    print(f"SimpleUpsamplerSubpixel: in_channels: {in_channels_0}, out_channels: {out_channels_0}")

    in_channels_1 = in_channels_0 + passthrough_channels_0
    out_channels_1 = in_channels_0
    print(f"BottleneckBlock: in_channels: {in_channels_1}, out_channels: {out_channels_1}")

    down -= 1

print(f"modulator_sizes: {modulator_sizes}")
