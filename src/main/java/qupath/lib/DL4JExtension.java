package qupath.lib;

import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.commands.BrightnessContrastCommand;
import qupath.lib.gui.extensions.QuPathExtension;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;

public class DL4JExtension implements QuPathExtension {

    @Override
    public void installExtension(QuPathGUI qupath) {

        // Get reference to menu
        Menu menu = qupath.getMenu("Extensions>DL4J", true);

        // Add new items to the menu
        QuPathGUI.addMenuItems(
                menu,
                QuPathGUI.createCommandAction(new TestCommand(qupath), "Test")
        );

        // Experimental Non-plugin item
       /*
        MenuItem item = new MenuItem("Experimental");

        item.setOnAction(e -> {
            new BrightnessContrastCommand(qupath).run();
        });

        menu.getItems().add(item);
        */
    }
    @Override
    public String getName(){
        return "DL4J-extension";
    }

    @Override
    public String getDescription(){
        return "Extension for DL4J in QuPath.";
    }

}
