#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FReinBalanceEditorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override {}
	virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FReinBalanceEditorModule, ReinBalanceEditor)
