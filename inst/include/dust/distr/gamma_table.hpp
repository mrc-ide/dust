#ifndef DUST_DISTR_GAMMA_TABLE_HPP
#define DUST_DISTR_GAMMA_TABLE_HPP

#ifdef __NVCC__
  // Exact function for this table is
  // std::lgamma(k + 1) - (std::log(std::sqrt(2 * M_PI)) +
  //    (k + 0.5) * std::log(k + 1) - (k + 1))
__constant__ float cudakTailValues[] = {
                                  0.041340695955409457f,
                                  0.027677925684998161f,
                                  0.020790672103765173f,
                                  0.016644691189821259f,
                                  0.013876128823071099f,
                                  0.011896709945892425f,
                                  0.010411265261970115f,
                                  0.009255462182705898f,
                                  0.0083305634333594725f,
                                  0.0075736754879489609f,
                                  0.0069428401072073598f,
                                  0.006408994188007f,
                                  0.0059513701127578145f,
                                  0.0055547335519605667f,
                                  0.0052076559196052585f,
                                  0.0049013959484298653f,
                                  0.0046291537493345913f,
                                  0.0043855602492257617f,
                                  0.0041663196919898837f,
                                  0.0039679542186377148f,
                                  0.0037876180684577321f,
                                  0.0036229602246820036f,
                                  0.0034720213829828594f,
                                  0.0033331556367386383f,
                                  0.0032049702280616543f,
                                  0.0030862786826162392f,
                                  0.0029760639835672009f,
                                  0.0028734493623545632f,
                                  0.0027776749297458991f,
                                  0.0026880788285268409f,
                                  0.0026040819192587605f,
                                  0.0025251752497723601f,
                                  0.0024509097354297182f,
                                  0.0023808876082398456f,
                                  0.0023147552905129487f,
                                  0.0022521974243261411f,
                                  0.0021929318432967193f,
                                  0.0021367053177385742f,
                                  0.0020832899382980941f,
                                  0.0020324800282622846f,
                                  0.0019840894972418255f,
                                  0.0019379495639952893f,
                                  0.0018939067895757944f,
                                  0.0018518213729947774f,
                                  0.0018115656687029968f,
                                  0.0017730228939285553f,
                                  0.0017360859968675868f,
                                  0.0017006566641839527f,
                                  0.0016666444469990438f,
                                  0.001633965989896069f,
                                  0.0016025443491400893f,
                                  0.0015723083877219324f,
                                  0.0015431922375341856f,
                                  0.0015151348208348736f,
                                  0.0014880794221880933f,
                                  0.0014619733060499129f,
                                  0.001436767373576231f,
                                  0.0014124158545030241f,
                                  0.0013888760298357283f,
                                  0.0013661079815676658f,
                                  0.0013440743670685151f,
                                  0.0013227402145616907f,
                                  0.0013020727377295316f,
                                  0.001282041167968373f,
                                  0.0012626166012807971f,
                                  0.0012437718593503178f,
                                  0.0012254813623826522f,
                                  0.0012077210134293637f,
                                  0.0011904680924885724f,
                                  0.0011737011595869262f,
                                  0.0011573999656775413f,
                                  0.0011415453712970702f,
                                  0.0011261192715892321f,
                                  0.001111104527126372f,
                                  0.0010964849005574706f,
                                  0.0010822449980310012f,
                                  0.0010683702151936814f,
                                  0.0010548466869408912f,
                                  0.0010416612416292992f,
                                  0.0010288013577337551f,
                                  0.0010162551247958618f,
                                  0.0010040112064189088f,
                                  0.00099205880559338766f,
                                  0.0009803876338878581f,
                                  0.00096898788103771949f,
                                  0.00095785018794458665f,
                                  0.00094696562098306458f,
                                  0.00093632564789913886f,
                                  0.00092592211569808569f,
                                  0.00091574722972609379f,
                                  0.00090579353434350196f,
                                  0.00089605389439384453f,
                                  0.00088652147843504281f,
                                  0.00087718974270956096f,
                                  0.00086805241602405658f,
                                  0.00085910348576589968f,
                                  0.0008503371848291863f,
                                  0.00084174797910918642f,
                                  0.00083333055562206937f,
                                  0.00082507981221624505f,
                                  0.00081699084654474063f,
                                  0.00080905894668603651f,
                                  0.00080127958193543236f,
                                  0.00079364839422169098f,
                                  0.00078616118980789906f,
                                  0.00077881393195866622f,
                                  0.00077160273326626339f,
                                  0.00076452384899994286f,
                                  0.00075757367056894509f,
                                  0.00075074871966762657f,
                                  0.00074404564190899691f,
                                  0.00073746120165196771f,
                                  0.00073099227711281856f,
                                  0.00072463585468085512f,
                                  0.00071838902505305668f,
                                  0.0007122489779476382f,
                                  0.00070621299857975828f,
                                  0.0007002784636256365f,
                                  0.00069444283690245356f,
                                  0.00068870366612827638f,
                                  0.00068305857956829641f,
                                  0.0006775052823400074f,
                                  0.00067204155374156471f,
                                  0.00066666524440961439f,
                                  0.00066137427273815774f,
                                  0.00065616662294587513f,
                                  0.00065104034212026818f,
                                  0.00064599353800076642f};
#endif
const double kTailValues[] = {
                                  0.041340695955409457,
                                  0.027677925684998161,
                                  0.020790672103765173,
                                  0.016644691189821259,
                                  0.013876128823071099,
                                  0.011896709945892425,
                                  0.010411265261970115,
                                  0.009255462182705898,
                                  0.0083305634333594725,
                                  0.0075736754879489609,
                                  0.0069428401072073598,
                                  0.006408994188007,
                                  0.0059513701127578145,
                                  0.0055547335519605667,
                                  0.0052076559196052585,
                                  0.0049013959484298653,
                                  0.0046291537493345913,
                                  0.0043855602492257617,
                                  0.0041663196919898837,
                                  0.0039679542186377148,
                                  0.0037876180684577321,
                                  0.0036229602246820036,
                                  0.0034720213829828594,
                                  0.0033331556367386383,
                                  0.0032049702280616543,
                                  0.0030862786826162392,
                                  0.0029760639835672009,
                                  0.0028734493623545632,
                                  0.0027776749297458991,
                                  0.0026880788285268409,
                                  0.0026040819192587605,
                                  0.0025251752497723601,
                                  0.0024509097354297182,
                                  0.0023808876082398456,
                                  0.0023147552905129487,
                                  0.0022521974243261411,
                                  0.0021929318432967193,
                                  0.0021367053177385742,
                                  0.0020832899382980941,
                                  0.0020324800282622846,
                                  0.0019840894972418255,
                                  0.0019379495639952893,
                                  0.0018939067895757944,
                                  0.0018518213729947774,
                                  0.0018115656687029968,
                                  0.0017730228939285553,
                                  0.0017360859968675868,
                                  0.0017006566641839527,
                                  0.0016666444469990438,
                                  0.001633965989896069,
                                  0.0016025443491400893,
                                  0.0015723083877219324,
                                  0.0015431922375341856,
                                  0.0015151348208348736,
                                  0.0014880794221880933,
                                  0.0014619733060499129,
                                  0.001436767373576231,
                                  0.0014124158545030241,
                                  0.0013888760298357283,
                                  0.0013661079815676658,
                                  0.0013440743670685151,
                                  0.0013227402145616907,
                                  0.0013020727377295316,
                                  0.001282041167968373,
                                  0.0012626166012807971,
                                  0.0012437718593503178,
                                  0.0012254813623826522,
                                  0.0012077210134293637,
                                  0.0011904680924885724,
                                  0.0011737011595869262,
                                  0.0011573999656775413,
                                  0.0011415453712970702,
                                  0.0011261192715892321,
                                  0.001111104527126372,
                                  0.0010964849005574706,
                                  0.0010822449980310012,
                                  0.0010683702151936814,
                                  0.0010548466869408912,
                                  0.0010416612416292992,
                                  0.0010288013577337551,
                                  0.0010162551247958618,
                                  0.0010040112064189088,
                                  0.00099205880559338766,
                                  0.0009803876338878581,
                                  0.00096898788103771949,
                                  0.00095785018794458665,
                                  0.00094696562098306458,
                                  0.00093632564789913886,
                                  0.00092592211569808569,
                                  0.00091574722972609379,
                                  0.00090579353434350196,
                                  0.00089605389439384453,
                                  0.00088652147843504281,
                                  0.00087718974270956096,
                                  0.00086805241602405658,
                                  0.00085910348576589968,
                                  0.0008503371848291863,
                                  0.00084174797910918642,
                                  0.00083333055562206937,
                                  0.00082507981221624505,
                                  0.00081699084654474063,
                                  0.00080905894668603651,
                                  0.00080127958193543236,
                                  0.00079364839422169098,
                                  0.00078616118980789906,
                                  0.00077881393195866622,
                                  0.00077160273326626339,
                                  0.00076452384899994286,
                                  0.00075757367056894509,
                                  0.00075074871966762657,
                                  0.00074404564190899691,
                                  0.00073746120165196771,
                                  0.00073099227711281856,
                                  0.00072463585468085512,
                                  0.00071838902505305668,
                                  0.0007122489779476382,
                                  0.00070621299857975828,
                                  0.0007002784636256365,
                                  0.00069444283690245356,
                                  0.00068870366612827638,
                                  0.00068305857956829641,
                                  0.0006775052823400074,
                                  0.00067204155374156471,
                                  0.00066666524440961439,
                                  0.00066137427273815774,
                                  0.00065616662294587513,
                                  0.00065104034212026818,
                                  0.00064599353800076642f};

#endif