#pragma once

namespace kuiper_infer {
   enum class RuntimeParameterType {
    kParameterUnknown = 0,
    kParameterBool = 1,
    kParameterInt = 2,

    kParameterFloat = 3,
    kParameterString = 4,
    kParameterIntArray = 5,
    kParameterFloatArray = 6,
    kParameterStringArray = 7,
    };
}