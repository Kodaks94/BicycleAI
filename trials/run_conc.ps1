Set-Location -Path "C:\Users\Mahra\Desktop\Projects\Thesis\Chapter 4-5 ADP\bicycle\"
ls
$app1 = Start-Process  Powershell.exe -Argumentlist "-noexit -file .\run_experiment.ps1 -trial 1 -core 48 -goal 0 -tanh 1"

$app2 = Start-Process  Powershell.exe -Argumentlist "-noexit -file .\run_experiment.ps1 -trial 2 -core 96 -goal 0 -tanh 0"

$app3 = Start-Process Powershell.exe -Argumentlist "-noexit -file .\run_experiment.ps1 -trial 3 -core 192 -goal 1 -tanh 1"

$app4 = Start-Process Powershell.exe -Argumentlist "-noexit -file .\run_experiment.ps1 -trial 4 -core 384 -goal 1 -tanh 0"




