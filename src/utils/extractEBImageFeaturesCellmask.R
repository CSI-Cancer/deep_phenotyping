library('EBImage')

#### Usage
# Receive image and mask folder paths
image_path <- commandArgs(trailingOnly = TRUE)[1]
# outfile_path <- commandArgs(trailingOnly = TRUE)[2]

image = readImage(image_path)
dapi = image[,,1] * 65535
tritc = image[,,2] * 65535
cy5 = image[,,3] * 65535
fitc = image[,,4] * 65535
mask = image[,,5] * 65535

# Compute the features
features = as.data.frame(
	computeFeatures(
		x=mask,
		ref = list(dapi,tritc,cy5,fitc),
		methods.ref = c(
			"computeFeatures.basic",
			"computeFeatures.moment",
			"computeFeatures.haralick"
			),
		xname = "cellf",
		refnames = c("dapi", "tritc","cy5","fitc")
	)
)

# Add image names (frame_id) and cell_id to the features
frame_id = gsub(".tif", "", basename(image_path))
features$frame_id = as.integer(frame_id)
features$cell_id = seq(1, nrow(features))
# features = Map(function(df, str) {df$frame_id <- as.integer(str); df$cell_id <- seq(1, nrow(df)); df}, features, frame_ids)

# Save the features
write.table(features,file=paste0(frame_id,'.txt'),row.names=FALSE,quote=FALSE,sep="\t")
print(image_path)