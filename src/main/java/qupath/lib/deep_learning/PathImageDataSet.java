package qupath.lib.deep_learning;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import qupath.imagej.objects.PathImagePlus;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ServerTools;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;

public class PathImageDataSet extends PathDataSet {

    public PathImageDataSet (final PathObject p, INDArray features, INDArray labels) {
        super(p, features, labels);
    }

    public PathImageDataSet () {
        super();
    }

    /**
     * Convert the image data corresponding with the pathObject into a feature matrix
     *
     * TODO Rewrite this to not have to calculate all these things...
     */
    @Override
    protected void convertFeatures () {

        ImageData<BufferedImage> imageData = QuPathGUI.getInstance().getImageData();
        ImageServer<BufferedImage> server = imageData.getServer();
        double downsample = imageData.getServer().getAveragedPixelSizeMicrons();
        BufferedImage img = server.readBufferedImage(RegionRequest.createInstance(server.getPath(), downsample, pathObject.getROI()));

        INDArray array = Nd4j.create((DataBuffer) img.getData().getDataBuffer());

        logger.info(array.toString());
    }
}
