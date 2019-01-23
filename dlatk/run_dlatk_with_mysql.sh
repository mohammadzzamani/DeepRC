#!/bin/bash
counter=1
while [ $counter -le 10 ]
do
   python3.5 dlatkInterface_NN.py  -d county_disease -t msgs_100u -c cnty --group_freq_thresh 20000  -f    'feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$16to16'  --outcome_table  topDeaths_comp_10to15_nonnull_10  --outcomes   '01hea_aar'     --model ridgecv --folds 10 --combo_test_reg    --controls  'hsgradHC03_VC93ACS3yr$10' 'bachdegHC03_VC94ACS3yr$10'     'logincomeHC01_VC85ACS3yr$10'  'unemployAve_BLSLAUS$0910' 'femalePOP165210D$10'  'hispanicPOP405210D$10' 'blackPOP255210D$10'  'forgnbornHC03_VC134ACS3yr$10'  'county_density' 'marriedaveHC03_AC3yr$10' 'median_age'   --factor_addition --num_of_factors 11  --fold_column fold
   echo $counter
   mysql --user="mzamani"  --database="county_disease" --execute="update  topDeaths_comp_10to15_nonnull_10 set fold = mod((fold + 1), 10) ;"
   counter=$(( $counter + 1 ))
done
