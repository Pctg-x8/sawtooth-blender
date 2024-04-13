VertexInput { Binding 0 [PerVertex] { y: float; } }
VertexShader {
    const float x = (gl_VertexIndex / float(GenerateSamples)) * 2.0f - 1.0f;
    RasterPosition = vec4(x, -y * 0.8f, 0.0f, 1.0f);
}

FragmentShader {
    Target[0] = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

SpecConstant[VertexShader](0) GenerateSamples: uint = 44100;
