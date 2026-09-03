$ErrorActionPreference = 'Stop'
try {
  $official = @{v084=16517;v086=16744;v138=15715;v139=15716;v140=15838;v147=16579;v155=16581;v156=16580;v157=16729;v158=16861}
  $cache = @{}
  foreach ($v in @('v084','v086','v138','v139','v140','v147','v155','v156','v157','v158')) {
    $cache[$v] = (Get-Content "artifacts\official_eval\reeval5-$v-default.json" -Raw | ConvertFrom-Json).results[0]
  }
  $pairs = @(
    @{b='v084';c='v086'},@{b='v138';c='v139'},@{b='v138';c='v140'},@{b='v140';c='v147'},
    @{b='v086';c='v147'},@{b='v147';c='v155'},@{b='v147';c='v156'},@{b='v086';c='v157'}
  )
  foreach ($p in $pairs) {
    $od = $official[$p.c] - $official[$p.b]
    Write-Output ("=== {0} -> {1}  (official delta {2}) ===" -f $p.b,$p.c,$od)
    foreach ($side in @('linear','attention')) {
      $bIdx = @{}
      foreach ($cs in $cache[$p.b].case_scores.$side) {
        $bIdx[("$($cs.layer)|$($cs.role)|$($cs.test_window)|$($cs.test_split)|$($cs.test_length)")] = $cs
      }
      $deltas = New-Object System.Collections.Generic.List[double]
      $imp=0;$reg=0;$unch=0;$mm=0
      $byRole = @{}
      foreach ($cs in $cache[$p.c].case_scores.$side) {
        $bc = $bIdx[("$($cs.layer)|$($cs.role)|$($cs.test_window)|$($cs.test_split)|$($cs.test_length)")]
        if ($null -eq $bc) { $mm++; continue }
        if ([math]::Abs($bc.mse_standard - $cs.mse_standard) -gt 1e-9 * [math]::Max(1.0,[math]::Abs($bc.mse_standard))) { $mm++; continue }
        $d = $cs.gain - $bc.gain
        $deltas.Add($d)
        if ($d -gt 1e-9) { $imp++ } elseif ($d -lt -1e-9) { $reg++ } else { $unch++ }
        if ($side -eq 'linear') {
          if (-not $byRole.ContainsKey($cs.role)) { $byRole[$cs.role]=New-Object System.Collections.Generic.List[double] }
          $byRole[$cs.role].Add($d)
        }
      }
      if ($deltas.Count -eq 0) { continue }
      $mean = ($deltas | Measure-Object -Average).Average
      $s = $deltas | Sort-Object
      $median = $s[[int](($s.Count-1)/2)]
      $min = $s[0]; $max = $s[-1]
      $touch = [math]::Round(100*($imp+$reg)/$deltas.Count,1)
      Write-Output ("  {0}: n={1} meanD={2} medD={3} minD={4} maxD={5} imp/reg/unch={6}/{7}/{8} touch={9}% mism={10}" -f $side,$deltas.Count,($mean.ToString('+0.000000;-0.000000;0')),($median.ToString('+0.000000;-0.000000;0')),($min.ToString('+0.000000;-0.000000;0')),($max.ToString('+0.000000;-0.000000;0')),$imp,$reg,$unch,$touch,$mm)
      if ($side -eq 'linear' -and ($imp+$reg) -gt 0) {
        $rl = @()
        foreach ($k in ($byRole.Keys | Sort-Object)) {
          $rm = ($byRole[$k] | Measure-Object -Average).Average
          $ri = @($byRole[$k] | Where-Object {$_ -gt 1e-9}).Count
          $rr = @($byRole[$k] | Where-Object {$_ -lt -1e-9}).Count
          $rl += ("{0}:{1}({2}+/{3}-)" -f $k,$rm.ToString('+0.0000;-0.0000;0'),$ri,$rr)
        }
        Write-Output ("    byRole: " + ($rl -join '  '))
      }
    }
  }
} catch {
  Write-Output ("ERROR: " + $_.Exception.Message)
  Write-Output ($_.ScriptStackTrace)
}
