<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
	<xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>

	<xsl:template match="@*|node()">
		<xsl:copy>
		    <xsl:apply-templates select="@*|node()"/>
		</xsl:copy>
	</xsl:template>

	<xsl:template match="row/*">
	    <field>
    	    <xsl:attribute name="field">
    	        <xsl:value-of select="name()"/>
    	    </xsl:attribute>
    	    <xsl:value-of select="text()"/>
	    </field>
	</xsl:template>
</xsl:stylesheet>
