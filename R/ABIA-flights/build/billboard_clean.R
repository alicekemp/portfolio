billboard = read_csv(here(("data/billboard.csv"))) %>%
  select(performer, song, year, week, week_position) %>%
write_csv(here("data/billboard_clean.csv"))