# June 19th 2019
suppressPackageStartupMessages({
  library("purrr")
  library("readr")
  library("dplyr")
  library("tidyr")
  library("reldist")
})

folder_in <- "data"
#anomalies <- c("Ponzi", "hacks", "RWare", "random", "realrandom", "LSeed")
anomalies <- c("Ponzi")

### local functions

summary_stats <- function(x, names_prefix = "") {
  res <- data.frame(sum = sum(x),
                    mean = mean(x),
                    sd = sd(x, na.rm =TRUE),
                    gini = gini(x),
                    degree = length(x))
  names(res) <- paste0(names_prefix, names(res))
  res
}

delay_stats <- function(tstamp, direction, names_prefix = "") {
  res <- data.frame(tstamp = as.numeric(tstamp),
                    dir = direction) %>%
         arrange(tstamp) %>%
         unique() %>%
         mutate(tstamp_prev = lag(tstamp),
                dir_prev = lag(dir)) %>%
#  print(res)
#  xx <<- res
#  res <- res %>%
         filter(dir != dir_prev) %>%
         mutate(tstamp_diff = abs(tstamp - tstamp_prev)) %>%
         summarize(min = min(tstamp_diff),
                   max = max(tstamp_diff),
                   mean = mean(tstamp_diff))
         names(res) <- paste0(names_prefix, names(res))
  res
}

### read data sets

dat_inp <- map_df(anomalies,
                  function(x)
                    read_delim(file.path(folder_in,
                                         paste0("addresses_", x, "_tx.csv.gz")),
                               delim = ";", col_names = TRUE) %>%
                    mutate(type = x,
                           date = as.Date(timestamp),
                           value = value / 1e8))

dat_ioaddr <- map_df(anomalies,
                     function(x)
                       read_delim(file.path(folder_in,
                                           paste0("addresses_", x,
                                                   "_ioaddr.csv.gz")),
                                  delim = ";", col_names = TRUE,
                                  col_types = "cic") %>%
                      mutate(type = x))

### compute features

stats_in <- dat_inp %>%
            filter(value >= 0) %>%
            group_by(type, address) %>%
            do(stats = summary_stats(.$value, "in_")) %>%
            unnest()

stats_out <- dat_inp %>%
             filter(value < 0) %>%
             group_by(type, address) %>%
             do(stats = summary_stats(.$value, "out_")) %>%
             unnest()

stats_delay <- dat_inp %>%
               group_by(type, address) %>%
#               do(stats = delay_stats(.$value, .$value >= 0, "delay_")) %>%   # changed this !!!
                 do(stats = delay_stats(.$timestamp, .$value >= 0, "delay_")) %>%
               unnest()

stats_activity <- dat_inp %>%
                  group_by(type, address) %>%
                  summarize(lifetime = diff(range(as.numeric(timestamp))),
                            lifetime_days = diff(as.numeric(range(date))),
                            act_day = length(unique(date)))

max_tx_day <- dat_inp %>%
              group_by(type, address, date) %>%
              summarize(tx_per_day = n()) %>%
              group_by(type, address) %>%
              summarize(max_tx_per_day = max(tx_per_day))

dat <- list(stats_in, stats_out, stats_delay, stats_activity, max_tx_day) %>%
       reduce(left_join, c("type", "address")) %>%
       mutate(net_value = in_sum + out_sum,
              ratio_degree = in_degree / out_degree)

dat <- merge(dat, subset(dat_ioaddr, select=c("address", "num_addresses")),  by ="address")

# write_delim(dat, path = "address_features.csv.gz", delim = ";")
# end here
