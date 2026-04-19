#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FPythonTrainingCommModule : public IModuleInterface
{
public:
	virtual void StartupModule() override {}
	virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FPythonTrainingCommModule, PythonTrainingComm)
