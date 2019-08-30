#ifndef TIMER_H
#define TIMER_H
#include <ctime>

class Timer
{
	clock_t startedAt;
	clock_t pausedAt;
	bool started;
	bool paused;
public:
	Timer();
	bool IsStarted();
	bool IsStopped();
	bool IsPaused();
	bool IsActive();
	void Pause();
	void Resume();
	void Stop();
	void Start();
	void Reset();
	clock_t GetTicks();
};
Timer::Timer()
{
	startedAt = 0;
	pausedAt = 0;
	paused = false;
	started = false;
}
bool Timer::IsStarted()
{
	return started;
}
bool Timer::IsStopped()
{
	return !started;
}
bool Timer::IsPaused()
{
	return paused;
}
bool Timer::IsActive()
{
	return !paused & started;
}
void Timer::Pause()
{
	if (paused || !started)
		return;
	paused = true;
	pausedAt = clock();
}
void Timer::Resume()
{
	if (!paused)
		return;
	paused = false;
	startedAt += clock() - pausedAt;
}
void Timer::Stop()
{
	started = false;
}
void Timer::Start()
{
	if (started)
		return;
	started = true;
	paused = false;
	startedAt = clock();
}
void Timer::Reset()
{
	paused = false;
	startedAt = clock();
}
clock_t Timer::GetTicks()
{
	if (!started)
		return 0;
	if (paused)
		return pausedAt - startedAt;
	
	return clock() - startedAt;
}


#endif