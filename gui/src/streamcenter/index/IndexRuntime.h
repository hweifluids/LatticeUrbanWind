#pragma once

#include <QString>

namespace Streamcenter::Index {

struct RuntimeStatus {
    bool available = false;
    QString message;
};

class IndexRuntime {
public:
    static RuntimeStatus status();
};

}  // namespace Streamcenter::Index
