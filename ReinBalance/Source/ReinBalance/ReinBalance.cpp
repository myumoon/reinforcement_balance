// Copyright Epic Games, Inc. All Rights Reserved.

#include "ReinBalance.h"
#include "Modules/ModuleManager.h"

#if IS_PROGRAM
IMPLEMENT_GAME_MODULE(FDefaultGameModuleImpl, ReinBalance);
#else
IMPLEMENT_PRIMARY_GAME_MODULE(FDefaultGameModuleImpl, ReinBalance, "ReinBalance");
#endif
