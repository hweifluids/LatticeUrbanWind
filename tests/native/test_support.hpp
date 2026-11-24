#pragma once

#include <QString>

#include <iostream>

struct TestContext {
    int failures = 0;

    void expect(bool condition, const QString& message) {
        if (condition) return;
        ++failures;
        std::cerr << "FAIL: " << message.toStdString() << '\n';
    }

    template <typename Actual, typename Expected>
    void equal(const Actual& actual, const Expected& expected, const QString& message) {
        expect(actual == expected, message);
    }
};

void runTerrainVoxelConfigTests(TestContext& context);

