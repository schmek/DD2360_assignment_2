#pragma once
#include <chrono>
#include <string>

class ProfilingTimer
{
public:
  using clock_t = std::chrono::steady_clock;
  ProfilingTimer(const std::string& description) 
    : m_desc(description)
    , m_start(clock_t::now())
  {};

  ProfilingTimer(const ProfilingTimer&) = delete;
  ProfilingTimer& operator=(const ProfilingTimer&) = delete;

  float lap_time()
  {
    using fdur = std::chrono::duration<float, std::milli>;
    fdur ms = clock_t::now() - m_start;
    m_lastLapTime = ms.count();
    return ms.count();
  }

  ~ProfilingTimer()
  {
    if (m_lastLapTime == 0.f)
      lap_time();
    printf("Time spent in %s: %.3f ms\n", m_desc.c_str(), m_lastLapTime);
  }

private:
  const std::string m_desc;
  clock_t::time_point m_start; //Important! m_start should be last so that the string ctor is not counted.
  float m_lastLapTime = 0.f;
};