param(
[Parameter()]
[Int64]$core,
[String]$trial,
[int64]$goal,
[int64]$tanh
)
echo "running on  $core  core"
Set-Location -Path "C:\Users\Mahra\Desktop\Projects\Thesis\Chapter 4-5 ADP\bicycle"
conda activate tf
python Original_bike.py --trialname=$trial --with_psi_restriction=1 --goal=$goal --use_tanh=$tanh --core=$core


