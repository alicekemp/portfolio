if (!("librarian" %in% rownames(utils::installed.packages()))) {
  utils::install.packages("librarian")
}
librarian::shelf(
  tidyverse,
  dplyr,
  haven, 
  stargazer, 
  RColorBrewer,
  here,
  ggmap,
  scales, 
  ggridges,
  viridis,
  gganimate, 
  gifski, 
  future, 
  geosphere,
  maps, 
  airportr, 
  kable, 
  kableExtra)
## color palette brewing
plasma = c("darkorange2", "coral2", "orangered2", "violetred2", "magenta4", "purple3", "mediumblue")

here::i_am("R/include.R")
