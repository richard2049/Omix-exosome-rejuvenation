## process_OMIX007582_Mammal40.R
## Process Mammal40 IDAT files for OMIX007582 into a beta-value matrix.
## Outputs:
##   1) OMIX007582_beta_matrix.csv (CpGs x samples, technical IDs)
##   2) OMIX007582_beta_matrix_named_partial.csv (OPTIONAL, partially mapped, see limitations)
##
## NOTE: Metadata table OMIX007582-02.csv has 620 rows, beta matrix has 643 columns.
##       There is no 1:1 sample match available; any mapping based on order is approximate
##       and should be treated as exploratory.

## 1) Load / install required packages ---------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

needed_pkgs <- c("sesame", "sesameData", "BiocParallel")
for (pkg in needed_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

library(sesame)
library(sesameData)
library(BiocParallel)

## 2) Cache sesameData (annotation & resources) ------------------------
try({
  sesameDataCache()
}, silent = TRUE)

## 3) Force serial execution to avoid BiocParallel issues --------------
BPPARAM <- SerialParam()

## 4) Folder containing your IDAT files --------------------------------
base_dir <- "D:/DATA/SRSC/data/OMIX007582_idat"

if (!dir.exists(base_dir)) {
  stop("Base directory does not exist: ", base_dir)
}

## 5) Detect IDAT prefixes (one prefix per sample) ---------------------
idat_files <- list.files(
  base_dir,
  pattern = "[Rr]ed\\.idat$|[Gg]rn\\.idat$",
  full.names = FALSE
)

if (length(idat_files) == 0) {
  stop("No IDAT files found in: ", base_dir)
}

prefixes <- unique(sub("_[Rr]ed\\.idat$|_[Gg]rn\\.idat$", "", idat_files))

cat("Found", length(prefixes), "IDAT prefixes:\n")
print(head(prefixes))

## 6) Helper to process a single sample prefix -------------------------
process_one_sample <- function(prefix_path, BPPARAM, debug_save = FALSE) {
  prep_candidates <- c("SHCDPM", NA)

  for (prep in prep_candidates) {
    cat("  Trying prep =", ifelse(is.na(prep), "<default>", prep), "...\n")
    bet <- tryCatch(
      {
        if (is.na(prep)) {
          openSesame(prefix_path, BPPARAM = BPPARAM)
        } else {
          openSesame(prefix_path, prep = prep, BPPARAM = BPPARAM)
        }
      },
      error = function(e) {
        warning("    openSesame error for ", prefix_path,
                " with prep=",
                ifelse(is.na(prep), "<default>", prep),
                ": ", conditionMessage(e))
        NULL
      }
    )

    if (is.null(bet)) {
      next
    }

    cat("    Class of bet:", paste(class(bet), collapse = ", "), "\n")

    if (debug_save) {
      saveRDS(bet, file = file.path(base_dir, "debug_one_sample_bet.rds"))
      cat("    Saved bet object for debugging to debug_one_sample_bet.rds\n")
    }

    vec <- NULL
    if (is.numeric(bet)) {
      vec <- bet
    } else if (!is.null(bet$beta)) {
      vec <- bet$beta
    } else if (!is.null(bet$betas)) {
      vec <- bet$betas
    }

    if (!is.null(vec) && length(vec) > 0) {
      attr(vec, "prep_used") <- prep
      cat("    Obtained", length(vec), "beta values.\n")
      return(vec)
    } else {
      warning("    Non-empty bet object but no usable beta vector; trying next prep.")
    }
  }

  warning("  No usable beta vector produced for ", prefix_path, ".")
  return(NULL)
}

## 7) Process all sample prefixes --------------------------------------
beta_list <- list()
first_debug_saved <- FALSE

for (p in prefixes) {
  prefix_path <- file.path(base_dir, p)
  cat("Processing sample prefix:", prefix_path, "...\n")

  bet_vec <- process_one_sample(
    prefix_path,
    BPPARAM = BPPARAM,
    debug_save = !first_debug_saved
  )

  if (!is.null(bet_vec)) {
    beta_list[[p]] <- bet_vec
    first_debug_saved <- TRUE
  }
}

## 8) Sanity checks before building the matrix -------------------------
if (length(beta_list) == 0) {
  stop(
    "No valid beta vectors were produced for any sample.\n",
    "Likely causes:\n",
    "  - openSesame not correctly configured for this array,\n",
    "  - IDAT prefixes do not match expected naming,\n",
    "  - or the IDATs are not Mammal40/Infinium data SeSAMe can handle.\n",
    "Check debug_one_sample_bet.rds in the IDAT folder if it exists."
  )
}

lens <- vapply(beta_list, length, integer(1))
if (length(unique(lens)) != 1) {
  stop(
    "Samples have different beta vector lengths: ",
    paste(names(lens), lens, collapse = "; "),
    "\nThis usually indicates mixed platforms or corrupted IDATs."
  )
}

## 9) Build CpG x sample beta matrix -----------------------------------
beta_mat <- do.call(cbind, beta_list)
colnames(beta_mat) <- names(beta_list)

cat(
  "Beta matrix dimensions:",
  nrow(beta_mat), "CpGs x",
  ncol(beta_mat), "samples\n"
)
out_path_tech <- "D:/DATA/SRSC/data/OMIX007582_beta_matrix.csv"
write.csv(beta_mat, out_path_tech, row.names = TRUE)
cat("Technical-ID beta matrix written to:", out_path_tech, "\n")

## 10) OPTIONAL: partial renaming of columns using metadata order -------
## This is exploratory and assumes column order ≈ metadata row order.
## Disabled by default to avoid overstating sample alignment.

enable_partial_mapping <- FALSE  # <-- For exploratory mode "use enable_partial_mapping <- TRUE"

meta_path <- "D:/DATA/SRSC/data/OMIX007582-02.csv"
if (file.exists(meta_path) && enable_partial_mapping) {
  meta <- read.csv(meta_path, stringsAsFactors = FALSE)
  cat("Metadata rows:", nrow(meta), "\n")
  cat("Matrix columns:", ncol(beta_mat), "\n")

  if (nrow(meta) < ncol(beta_mat)) {
    n_map <- nrow(meta)

    if ("OriginalSampleName" %in% colnames(meta)) {
      new_names <- meta$OriginalSampleName
    } else if ("OriginalSampleName.1" %in% colnames(meta)) {
      new_names <- meta$OriginalSampleName.1
    } else {
      stop("No OriginalSampleName column found in metadata.")
    }

    warning(
      "Partial mapping enabled: renaming the first ", n_map,
      " columns using metadata order, leaving the remaining ",
      ncol(beta_mat) - n_map, " with technical IDs.\n",
      "This is an approximate mapping and MUST be reported as exploratory."
    )

    colnames(beta_mat)[seq_len(n_map)] <- new_names

    out_path_named <- "D:/DATA/SRSC/data/OMIX007582_beta_matrix_named_partial.csv"
    write.csv(beta_mat, out_path_named, row.names = TRUE)
    cat("Partially named beta matrix written to:", out_path_named, "\n")
  } else {
    warning(
      "Partial mapping requested but nrow(meta) >= ncol(beta_mat). ",
      "Not performing mapping."
    )
  }
} else {
  cat("Partial mapping disabled or metadata file not found; ",
      "only technical-ID matrix has been written.\n")
}

## 11) Save the matrix to CSV ------------------------------------------
out_path <- "D:/DATA/SRSC/data/OMIX007582_beta_matrix_named.csv"
write.csv(beta_mat, out_path, row.names = TRUE)

cat("Beta matrix written to:", out_path, "\n")