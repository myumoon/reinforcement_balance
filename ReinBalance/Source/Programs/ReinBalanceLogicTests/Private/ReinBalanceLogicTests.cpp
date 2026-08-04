// Copyright Epic Games, Inc. All Rights Reserved.

#include "HAL/PlatformMisc.h"
#include "Misc/CommandLine.h"
#include "TestCommon/Initialization.h"

#include <catch2/catch_test_macros.hpp>

GROUP_BEFORE_GLOBAL(Catch::DefaultGroup)
{
	FCommandLine::Append(TEXT(" -nullrhi -unattended"));
	FPlatformMisc::EngineDir();
	InitAll(/*bAllowLogging=*/true, /*bMultithreaded=*/true);
}

GROUP_AFTER_GLOBAL(Catch::DefaultGroup)
{
	CleanupAll();
}
