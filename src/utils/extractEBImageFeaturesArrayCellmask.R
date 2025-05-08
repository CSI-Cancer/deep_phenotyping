library('EBImage')

#### Usage
# Receive image file name
image_name <- commandArgs(trailingOnly = TRUE)[1]
output_name <- commandArgs(trailingOnly = TRUE)[2]
images <- readImage(image_name)
images <- Image(images, dim=c(dim(images)[1], dim(images)[2], 5, dim(images)[3] / 5))

##############
# define a function
get_features <- function(image){
	# Load the images and the masks
	dapi = image[,,1] * 65535
	tritc = image[,,2] * 65535
	cy5 = image[,,3] * 65535
	fitc = image[,,4] * 65535
	mask = image[,,5] * 65535
	# dapimask = thresh(normalize(image[,,1]), w=15, h=15, offset=0.05)
	# dapimask = opening(dapimask, makeBrush(5, shape='Gaussian'))
	# dapimask = fillHull(dapimask) * mask

	# nucleusfeatures_a = as.data.frame(
	# 	computeFeatures(
	# 		x=dapimask,
	# 		ref = list(dapi),
	# 		methods.ref = c(
	# 			"computeFeatures.basic",
	# 			"computeFeatures.moment",
	# 			"computeFeatures.haralick"
	# 			),
	# 		xname = "nucleusf",
	# 		refnames = c("dapi")
	# 	)
	# )

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

	# if(nrow(cellfeatures_a) == 0){
	# 	print("No cell features")
	# 	cellfeatures_a = as.data.frame(matrix(0, nrow=1, ncol=791))
	# }
	# if(nrow(nucleusfeatures_a) == 0){
	# 	print("No nucleus features")
	# 	nucleusfeatures_a = as.data.frame(matrix(0, nrow=1, ncol=89))
	# }
	# features = as.data.frame(cbind(as.matrix(cellfeatures_a),as.matrix(nucleusfeatures_a)))

	# Save the features
	return(features)
}

# Apply get_features on list of images
features = apply(images, MARGIN=4, get_features)

# synchronize column names
feature_names = colnames(features[[length(features)]])
feature_rename <- function(x, feature_names=feature_names){
	if (length(colnames(x)) == length(feature_names)){
		colnames(x) = feature_names
	}else{
		print(x)
	}
	return(x)
}
features = lapply(features, feature_rename, feature_names=feature_names)

# Combine the features
features = do.call(rbind, features)

# features$nucleus_cell_dist = ((features$nucleusf.0.m.cx - features$cellf.0.m.cx)^2 + (features$nucleusf.0.m.cy - features$cellf.0.m.cy)^2)^0.5
# features$cell_nucleus_ratio = with(features, cellf.0.s.area/nucleusf.0.s.area)
# features$tritc_cy5_ratio = with(features, cellf.tritc.b.q05/cellf.cy5.b.q05)
# features$cell_nucleus_ratio[features$cell_nucleus_ratio == Inf] = 10
# features$tritc_cy5_ratio[features$tritc_cy5_ratio == Inf] = 10
# features$cell_nucleus_ratio[is.na(features$cell_nucleus_ratio)] = 0
# features$tritc_cy5_ratio[is.na(features$tritc_cy5_ratio)] = 0
# features$image_name = image_name
print(dim(features))

# Save the features
write.table(features, file=output_name, row.names=FALSE, quote=FALSE, sep="\t")