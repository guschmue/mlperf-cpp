#pragma once

#include <memory>
#include <ostream>
#include <string>

namespace mlperf_bench {

enum StatusCategory {
    NONE = 0,
    SYSTEM = 1,
    RUNTIME = 2,
};

enum StatusCode {
    OK = static_cast<unsigned int>(0),
    FAIL = static_cast<unsigned int>(1),
};
      
class Status {
public:
 Status() noexcept = default;

 Status(StatusCategory category, int code, const std::string& msg);

 Status(StatusCategory category, int code, const char* msg);

 Status(StatusCategory category, int code);

 Status(const Status& other)
 : state_((other.state_ == nullptr) ? nullptr : std::make_unique<State>(*other.state_)) {}

Status& operator=(const Status& other) {
if (state_ != other.state_) {
 if (other.state_ == nullptr) {
   state_.reset();
 } else {
   state_ = std::make_unique<State>(*other.state_);
 }
}
return *this;
}

Status(Status&& other) = default;
Status& operator=(Status&& other) = default;
~Status() = default;

bool IsOK() const {
return (state_ == nullptr);
}

int Code() const noexcept;

StatusCategory Category() const noexcept;

const std::string& ErrorMessage() const noexcept;

std::string ToString() const;

bool operator==(const Status& other) const {
return (this->state_ == other.state_) || (ToString() == other.ToString());
}

bool operator!=(const Status& other) const {
return !(*this == other);
}

static Status OK() {
return Status();
}

private:
static const std::string& EmptyString() noexcept;

struct State {
    State(StatusCategory cat0, int code0, const std::string& msg0)
        : category(cat0), code(code0), msg(msg0) {}

    State(StatusCategory cat0, int code0, const char* msg0)
        : category(cat0), code(code0), msg(msg0) {}

    const StatusCategory category;
    const int code;
    const std::string msg;
};

// As long as Code() is OK, state_ == nullptr.
std::unique_ptr<State> state_;

};

}
