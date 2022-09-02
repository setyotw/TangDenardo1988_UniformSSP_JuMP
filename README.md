# Job Sequencing and Tool Switching Problem
This repository provides an implementation of Tang-Denardo formulation of job sequencing and tool switching problem (SSP) with uniform setup time. 
This formulation was introduced by [Tang and Denardo (1988)](https://doi.org/10.1287/opre.36.5.767). 

## Usage and Dependencies
* Codes are written in Julia using [JuMP framework](https://doi.org/10.1137/15M1020575).
* Solved with [Gurobi optimization solver](https://gurobi.com/) (under an academic license).

## Related works
This is the first mathematical formulation available for SSP, consider to check also: 
* [Laporte, Salazar-Gonzales, Semet (2004)](https://doi.org/10.1080/07408170490257871)
* [Catanzaro, Gouveia, Labbe (2015)](https://doi.org/10.1016/j.ejor.2015.02.018)
* [Da Silva, Chaves, Yanasse (2021)](https://doi.org/10.1080/00207543.2020.1748906)
* [Mara, Sutoyo, Norcahyo, Rifai (2021)](https://doi.org/10.1016/j.jksues.2021.02.015)
* [Rifai, Mara, Norcahyo (2022)](https://doi.org/10.1016/j.cie.2021.107813)

## License
This software is licensed under the MIT License. See file LICENSE for more information.
