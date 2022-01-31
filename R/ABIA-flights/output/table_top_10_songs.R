billboard_top = read_csv(here("data/billboard_clean.csv")) %>%
  group_by(performer, song) %>%
  filter(year >= 1958) %>%
  summarise(count = n()) %>%
  arrange(desc(count), .by_group = FALSE) 
billboard_top %>%
  slice_head(n = 10, .by_group=FALSE) %>%
kbl(
    caption = "Top 10 Billboard Songs since 1958",
    booktabs = TRUE,
    format = "latex",
    label = "tab:summarystats"
  ) %>% 
  kable_styling(latex_options = c("striped", "HOLD_position")) %>%
write_lines(here("tables/top_10_songs.tex"))
