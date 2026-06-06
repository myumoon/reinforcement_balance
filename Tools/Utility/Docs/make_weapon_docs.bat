@echo off

set OUT_DIR=..\..\..\.claude\docs\survivors_spec\weapons

call :MAKE_WEAPON_DOC whip
call :MAKE_WEAPON_DOC magic_wand
call :MAKE_WEAPON_DOC knife
call :MAKE_WEAPON_DOC axe
call :MAKE_WEAPON_DOC cross
call :MAKE_WEAPON_DOC king_bible
call :MAKE_WEAPON_DOC fire_wand
call :MAKE_WEAPON_DOC garlic
call :MAKE_WEAPON_DOC santa_water
call :MAKE_WEAPON_DOC runetracer
call :MAKE_WEAPON_DOC lightning_ring
call :MAKE_WEAPON_DOC pentagram
call :MAKE_WEAPON_DOC peachone
call :MAKE_WEAPON_DOC ebony_wings
call :MAKE_WEAPON_DOC phiera_der_tuphello
call :MAKE_WEAPON_DOC eight_the_sparrow
call :MAKE_WEAPON_DOC gatti_amari
call :MAKE_WEAPON_DOC song_of_mana
call :MAKE_WEAPON_DOC shadow_pinion
call :MAKE_WEAPON_DOC clock_lancet
call :MAKE_WEAPON_DOC laurel
call :MAKE_WEAPON_DOC vento_sacro
call :MAKE_WEAPON_DOC bone
call :MAKE_WEAPON_DOC cherry_bomb
call :MAKE_WEAPON_DOC2 Carréllo carrello
call :MAKE_WEAPON_DOC celestial_dusting
call :MAKE_WEAPON_DOC la_robba
call :MAKE_WEAPON_DOC greatest_jubilee
call :MAKE_WEAPON_DOC bracelet
call :MAKE_WEAPON_DOC candybox
call :MAKE_WEAPON_DOC victory_sword
call :MAKE_WEAPON_DOC flames_of_misspell
call :MAKE_WEAPON_DOC pako_battiliar
call :MAKE_WEAPON_DOC ammo_appalate
call :MAKE_WEAPON_DOC chaos_rune
call :MAKE_WEAPON_DOC glass_fandango
call :MAKE_WEAPON_DOC santa_javelin
call :MAKE_WEAPON_DOC gaze_of_gaea
call :MAKE_WEAPON_DOC magi-stone
call :MAKE_WEAPON_DOC phas3r
call :MAKE_WEAPON_DOC arma_dio

: Evolution
call :MAKE_WEAPON_DOC bloody_tear
call :MAKE_WEAPON_DOC holy_wand
call :MAKE_WEAPON_DOC thousand_edge
call :MAKE_WEAPON_DOC death_spiral
call :MAKE_WEAPON_DOC heaven_sword
call :MAKE_WEAPON_DOC unholy_vespers
call :MAKE_WEAPON_DOC hellfire
call :MAKE_WEAPON_DOC soul_eater
call :MAKE_WEAPON_DOC la_borra
call :MAKE_WEAPON_DOC no_future
call :MAKE_WEAPON_DOC thunder_loop
call :MAKE_WEAPON_DOC gorgeous_moon
call :MAKE_WEAPON_DOC vandalier
call :MAKE_WEAPON_DOC phieraggi
call :MAKE_WEAPON_DOC vicious_hunger
call :MAKE_WEAPON_DOC mannajja
call :MAKE_WEAPON_DOC valkyrie_turner
call :MAKE_WEAPON_DOC infinite_corridor
call :MAKE_WEAPON_DOC crimson_shroud
call :MAKE_WEAPON_DOC fuwalafuwaloo
call :MAKE_WEAPON_DOC anima_of_mortaccio
call :MAKE_WEAPON_DOC yatta_daikarin
call :MAKE_WEAPON_DOC2 Carozza! carozza

pause
goto :eof

:MAKE_WEAPON_DOC
call :MAKE_WEAPON_DOC2 %1 %1
exit /b 0

:MAKE_WEAPON_DOC2
python .\make_survivors_doc_weapon.py https://vampire.survivors.wiki/w/%1 %OUT_DIR%\weapon_%2.md
exit /b 0


